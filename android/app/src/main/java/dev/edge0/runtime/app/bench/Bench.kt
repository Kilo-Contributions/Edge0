// bench/Bench.kt - the benchmark protocol: tiers, instrumentation arguments, the fixed prompt,
// the measurement runner and its results, and model lookup. Depends only on the Runtime contract.
package dev.edge0.runtime.app.bench

import dev.edge0.runtime.app.runtime.GenEvent
import dev.edge0.runtime.app.runtime.RequestMetrics
import dev.edge0.runtime.app.runtime.Runtime
import dev.edge0.runtime.engine.ChatMessage
import dev.edge0.runtime.engine.GenParams
import java.io.File
import kotlinx.coroutines.delay

enum class BenchTier(val id: String, val modelFile: String) {
    EDGE0_8B("edge0-8b", "edge0-8b.gguf"),
    EDGE0_35B("edge0-35b", "edge0-35b.gguf");

    companion object {
        /** Accepts "8b", "35b" or the full "edge0-<size>" id, case-insensitive. */
        fun parse(raw: String?): BenchTier? {
            val key = raw?.trim()?.lowercase()?.removePrefix("edge0-") ?: return null
            return entries.firstOrNull { it.id.removePrefix("edge0-") == key }
        }
    }
}

/** Instrumentation arguments (`am instrument -e <key> <value>`). */
data class BenchArgs(val tier: BenchTier, val reporter: String, val notes: String, val temperature: Float) {
    companion object {
        const val KEY_TIER = "tier"
        const val KEY_REPORTER = "reporter"
        const val KEY_NOTES = "notes"
        const val KEY_TEMPERATURE = "temperature"

        /** The app's own default (Settings), so the benchmark samples like a normal chat turn. */
        const val DEFAULT_TEMPERATURE = 0.2f
        private const val MAX_TEMPERATURE = 2f

        /** @throws IllegalArgumentException when `tier` is missing or invalid, or `temperature` is out of range. */
        fun parse(args: Map<String, String?>): BenchArgs {
            val tier = BenchTier.parse(args[KEY_TIER])
                ?: throw IllegalArgumentException(
                    "-e $KEY_TIER must be 8b or 35b (got '${args[KEY_TIER].orEmpty()}')")
            val reporter = args[KEY_REPORTER].orEmpty().trim()
            val rawTemperature = args[KEY_TEMPERATURE].orEmpty().trim()
            val temperature = if (rawTemperature.isEmpty()) DEFAULT_TEMPERATURE else
                rawTemperature.toFloatOrNull()?.takeIf { it in 0f..MAX_TEMPERATURE }
                    ?: throw IllegalArgumentException(
                        "-e $KEY_TEMPERATURE must be a number from 0 (greedy) to $MAX_TEMPERATURE (got '$rawTemperature')")
            return BenchArgs(
                tier = tier,
                reporter = if (reporter.isEmpty() || reporter.startsWith("@")) reporter else "@$reporter",
                notes = args[KEY_NOTES].orEmpty().trim(),
                temperature = temperature,
            )
        }
    }
}

// Fixed long prompt so prefill numbers are comparable across devices. Paragraphs match
// python/examples/bench.py (BENCH_LONG). The instruction asks for a long answer because the
// engine stops at end-of-sequence and the decode window needs enough tokens.
private const val PARAGRAPH_COUNT = 40
private const val INSTRUCTION =
    "请仔细阅读以下材料，并写一篇不少于八百字的详细总结，分五个部分，每个部分至少一个完整段落："

internal const val PARAGRAPH_ZH =
    "流式推理框架的设计要点在于把磁盘带宽、缓存层级与计算核心三者重叠" +
    "起来。混合专家模型每一层只在少数专家上激活，路由器给出的稀疏选择" +
    "恰好为按需加载提供了天然的批粒度。若能在前一步的隐状态基础上预判" +
    "下一步的专家集合，固态硬盘的读取延迟就能被计算完全掩盖，这对边缘" +
    "设备上的大模型部署具有直接意义。"

internal const val PARAGRAPH_EN =
    "The design of a streaming inference framework hinges on overlapping " +
    "disk bandwidth, cache hierarchy, and compute. A mixture-of-experts " +
    "layer activates only a handful of experts per token, and the router " +
    "sparsity provides a natural granularity for on-demand loading. If " +
    "the expert set of the next step can be predicted from the previous " +
    "hidden state, SSD read latency hides behind compute entirely, which " +
    "matters for large-model deployment on edge hardware."

fun buildPrompt(): String = INSTRUCTION +
    List(PARAGRAPH_COUNT) { if (it % 2 == 0) PARAGRAPH_ZH else PARAGRAPH_EN }.joinToString(" ")

/** Finds the staged GGUF for a tier among candidate model roots; null entries are skipped. */
fun findModel(roots: List<File?>, tier: BenchTier): File? =
    roots.filterNotNull().map { File(it, tier.modelFile) }.firstOrNull { it.isFile }

data class BenchConfig(
    val reps: Int = 2,
    val maxNewTokens: Int = 200,
    val promptTokens: IntRange = 2_800..3_800,
    val pauseBetweenRepsMs: Long = 5_000,
    val temperature: Float = BenchArgs.DEFAULT_TEMPERATURE,
)

// The measurement protocol: one cold warmup (kept for the notes, not averaged), then N measured
// reps, each a fresh request (the runtime rebuilds the KV on a forked history, so every rep pays
// a full prefill). resetPeak clears the process peak-memory counter before each run so every
// run reports its own peak; it returns false when the platform refuses.
class BenchRunner(
    private val runtime: Runtime,
    private val config: BenchConfig = BenchConfig(),
    private val pause: suspend (Long) -> Unit = { delay(it) },
    private val resetPeak: () -> Boolean = { false },
) {
    /** @throws IllegalStateException when a turn fails or leaves the comparable measurement shape. */
    suspend fun run(prompt: String): BenchSummary {
        var peakPerRun = resetPeak()
        val cold = measure(prompt)
        val reps = List(config.reps) {
            pause(config.pauseBetweenRepsMs)
            peakPerRun = resetPeak() && peakPerRun
            measure(prompt)
        }
        return BenchSummary(reps, cold, peakPerRun)
    }

    private suspend fun measure(prompt: String): RepResult {
        var metrics: RequestMetrics? = null
        var failure: String? = null
        runtime.submit(
            THREAD_ID, listOf(ChatMessage("user", prompt)), thinkingOn = false,
            params = GenParams(temperature = config.temperature, maxNewTokens = config.maxNewTokens),
        ).collect { event ->
            when (event) {
                is GenEvent.Done -> metrics = event.metrics
                is GenEvent.Failed -> failure = event.diag
                is GenEvent.Cancelled -> failure = "cancelled"
                else -> Unit
            }
        }
        check(failure == null) { "generation failed: $failure" }
        return validated(checkNotNull(metrics) { "engine returned no metrics" })
    }

    private fun validated(metrics: RequestMetrics): RepResult {
        check(metrics.prefix_reused_tokens == 0) {
            "prefix cache reused ${metrics.prefix_reused_tokens} tokens; prefill would not be measured in full"
        }
        check(metrics.promptTokens in config.promptTokens) {
            "prompt is ${metrics.promptTokens} tokens, expected ${config.promptTokens}"
        }
        // every run must decode the full length: it measures an extended run, and the engine reports a
        // failed decode as a normal, shorter answer, which this check also catches
        check(metrics.newTokens == config.maxNewTokens) {
            "decode stopped after ${metrics.newTokens} tokens, need exactly ${config.maxNewTokens}"
        }
        return RepResult(
            promptTokens = metrics.promptTokens,
            newTokens = metrics.newTokens,
            prefillTokS = metrics.prefillTokS,
            decodeTokS = metrics.decodeTokS,
            peakRssBytes = metrics.peakRssBytes,
        )
    }

    private companion object { const val THREAD_ID = "bench" }
}

private const val BYTES_PER_GIB = (1L shl 30).toDouble()

data class RepResult(
    val promptTokens: Int,
    val newTokens: Int,
    val prefillTokS: Double,
    val decodeTokS: Double,
    val peakRssBytes: Long,
) {
    val peakGib: Double get() = peakRssBytes / BYTES_PER_GIB
}

/**
 * The block reports the mean of [reps]; [cold] is the discarded first run. [peakPerRun] is true when the
 * peak-memory counter was reset before every run; otherwise each peak is the whole-process maximum so far.
 */
data class BenchSummary(val reps: List<RepResult>, val cold: RepResult? = null, val peakPerRun: Boolean = false) {
    init { require(reps.isNotEmpty()) { "no measured reps" } }

    val prefillTokS: Double get() = reps.map { it.prefillTokS }.average()
    val decodeTokS: Double get() = reps.map { it.decodeTokS }.average()

    /** Peak resident set (VmHWM) in GiB, the highest of the measured runs. */
    val peakActiveGib: Double get() = reps.maxOf { it.peakGib }
}
