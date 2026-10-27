// BenchProtocolTest - argument and tier parsing, the fixed prompt (including fidelity to
// python/examples/bench.py), result aggregation, the measurement protocol and model lookup.
package dev.edge0.runtime.app.bench

import dev.edge0.runtime.app.runtime.GenEvent
import dev.edge0.runtime.app.runtime.GenState
import dev.edge0.runtime.app.runtime.RequestMetrics
import dev.edge0.runtime.app.runtime.Runtime
import dev.edge0.runtime.engine.ChatMessage
import dev.edge0.runtime.engine.GenParams
import dev.edge0.runtime.engine.Status
import java.io.File
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class BenchProtocolTest {

    @get:Rule val tmp = TemporaryFolder()

    // unit tests run with the module directory (android/app) as working directory
    private val pythonBench = File("../../python/examples/bench.py")

    private fun pythonParagraph(source: String, name: String): String {
        val body = Regex("$name = \\((.*?)\\)\\n", RegexOption.DOT_MATCHES_ALL)
            .find(source)!!.groupValues[1]
        return Regex("\"((?:[^\"\\\\]|\\\\.)*)\"").findAll(body).joinToString("") { it.groupValues[1] }
    }

    private fun rep(prefill: Double, decode: Double, peakBytes: Long) =
        RepResult(promptTokens = 3300, newTokens = 200, prefillTokS = prefill,
            decodeTokS = decode, peakRssBytes = peakBytes)

    private class ScriptedRuntime(private val turns: List<GenEvent>) : Runtime {
        override val state = MutableStateFlow<GenState>(GenState.Idle)
        override val lastMetrics = MutableStateFlow<RequestMetrics?>(null)
        override val activeModelDir: String? = "model"
        val submitted = mutableListOf<Triple<List<ChatMessage>, Boolean, GenParams>>()

        override suspend fun ensureLoaded(path: String) = Unit
        override suspend fun unloadActive() = Unit
        override suspend fun trim(ratio: Float) = true
        override fun cancelActive() = Unit
        override fun submit(threadId: String, messages: List<ChatMessage>,
                            thinkingOn: Boolean, params: GenParams): Flow<GenEvent> {
            submitted += Triple(messages, thinkingOn, params)
            return flowOf(turns[submitted.size - 1])
        }
    }

    private fun metrics(
        prompt: Int = 3300, reused: Int = 0, generated: Int = 200,
        prefill: Double = 100.0, decode: Double = 10.0, peak: Long = 1L shl 30,
    ) = RequestMetrics(
        promptTokens = prompt, prefix_reused_tokens = reused, newTokens = generated,
        prefillTokS = prefill, decodeTokS = decode, peakRssBytes = peak)

    private fun done(m: RequestMetrics = metrics()) = GenEvent.Done(m)

    private fun runner(runtime: Runtime, pauses: MutableList<Long> = mutableListOf()) =
        BenchRunner(runtime, BenchConfig(), pause = { pauses += it })

    @Test fun tierAcceptsShortAndFullNames() {
        assertEquals(BenchTier.EDGE0_8B, BenchTier.parse("8b"))
        assertEquals(BenchTier.EDGE0_8B, BenchTier.parse("edge0-8b"))
        assertEquals(BenchTier.EDGE0_35B, BenchTier.parse(" 35B "))
        assertEquals(BenchTier.EDGE0_35B, BenchTier.parse("Edge0-35b"))
    }

    @Test fun tierRejectsUnknownValues() {
        assertNull(BenchTier.parse("70b"))
        assertNull(BenchTier.parse(""))
        assertNull(BenchTier.parse(null))
    }

    @Test fun parsesAllArguments() {
        val args = BenchArgs.parse(mapOf("tier" to "35b", "reporter" to "@handle", "notes" to "case on"))
        assertEquals(BenchTier.EDGE0_35B, args.tier)
        assertEquals("@handle", args.reporter)
        assertEquals("case on", args.notes)
    }

    @Test fun reporterGetsHandlePrefix() {
        assertEquals("@handle", BenchArgs.parse(mapOf("tier" to "8b", "reporter" to "handle")).reporter)
    }

    @Test fun optionalArgumentsDefaultToEmpty() {
        val args = BenchArgs.parse(mapOf("tier" to "8b"))
        assertEquals("", args.reporter)
        assertEquals("", args.notes)
    }

    @Test fun invalidTierNamesTheAcceptedValues() {
        val error = assertThrows(IllegalArgumentException::class.java) {
            BenchArgs.parse(mapOf("tier" to "70b"))
        }
        assertEquals(true, error.message!!.contains("8b") && error.message!!.contains("35b"))
    }

    @Test fun missingTierIsRejected() {
        assertThrows(IllegalArgumentException::class.java) { BenchArgs.parse(emptyMap()) }
    }

    @Test fun temperatureDefaultsToTheAppSetting() {
        assertEquals(BenchArgs.DEFAULT_TEMPERATURE, BenchArgs.parse(mapOf("tier" to "8b")).temperature)
    }

    @Test fun parsesTemperature() {
        assertEquals(0f, BenchArgs.parse(mapOf("tier" to "8b", "temperature" to "0")).temperature)
        assertEquals(0.7f, BenchArgs.parse(mapOf("tier" to "8b", "temperature" to " 0.7 ")).temperature)
    }

    @Test fun rejectsAnOutOfRangeOrNonNumericTemperature() {
        listOf("abc", "-0.1", "2.5", "NaN").forEach { bad ->
            assertThrows(IllegalArgumentException::class.java) {
                BenchArgs.parse(mapOf("tier" to "8b", "temperature" to bad))
            }
        }
    }

    @Test fun promptIsDeterministic() {
        assertEquals(buildPrompt(), buildPrompt())
    }

    @Test fun promptAlternatesChineseAndEnglishParagraphs() {
        val prompt = buildPrompt()
        assertEquals(20, Regex("流式推理框架").findAll(prompt).count())
        assertEquals(20, Regex("The design of a streaming").findAll(prompt).count())
    }

    @Test fun paragraphsMatchThePythonBench() {
        assumeTrue("python/examples/bench.py not reachable from the test working directory", pythonBench.isFile)
        val source = pythonBench.readText()
        assertEquals(pythonParagraph(source, "_PARA_ZH"), PARAGRAPH_ZH)
        assertEquals(pythonParagraph(source, "_PARA_EN"), PARAGRAPH_EN)
    }

    @Test fun summaryAveragesThroughput() {
        val summary = BenchSummary(listOf(rep(100.0, 10.0, 1L shl 30), rep(120.0, 14.0, 1L shl 30)))
        assertEquals(110.0, summary.prefillTokS, 1e-9)
        assertEquals(12.0, summary.decodeTokS, 1e-9)
    }

    @Test fun summaryReportsLargestPeakInGib() {
        val summary = BenchSummary(listOf(rep(1.0, 1.0, 2L shl 30), rep(1.0, 1.0, 3L shl 30)))
        assertEquals(3.0, summary.peakActiveGib, 1e-9)
    }

    @Test fun emptySummaryIsRejected() {
        assertThrows(IllegalArgumentException::class.java) { BenchSummary(emptyList()) }
    }

    @Test fun discardsWarmupAndAveragesTheMeasuredReps() = runBlocking {
        val runtime = ScriptedRuntime(listOf(
            done(metrics(decode = 99.0)), done(metrics(decode = 10.0)), done(metrics(decode = 14.0))))
        val summary = runner(runtime).run("prompt")
        assertEquals(2, summary.reps.size)
        assertEquals(12.0, summary.decodeTokS, 1e-9)
    }

    @Test fun everyTurnIsAFreshThinkingOffRequest() = runBlocking {
        val runtime = ScriptedRuntime(List(3) { done() })
        runner(runtime).run("prompt")
        assertEquals(3, runtime.submitted.size)
        runtime.submitted.forEach { (messages, thinking, params) ->
            assertEquals(listOf(ChatMessage("user", "prompt")), messages)
            assertEquals(false, thinking)
            assertEquals(BenchArgs.DEFAULT_TEMPERATURE, params.temperature)
            assertEquals(200, params.maxNewTokens)
        }
    }

    @Test fun usesTheConfiguredTemperature() = runBlocking {
        val runtime = ScriptedRuntime(List(3) { done() })
        BenchRunner(runtime, BenchConfig(temperature = 0f), pause = {}).run("prompt")
        assertEquals(setOf(0f), runtime.submitted.map { it.third.temperature }.toSet())
    }

    @Test fun keepsTheColdRunApartFromTheMean() = runBlocking {
        val runtime = ScriptedRuntime(listOf(done(metrics(decode = 99.0)), done(), done()))
        val summary = runner(runtime).run("prompt")
        assertEquals(99.0, summary.cold!!.decodeTokS, 1e-9)
        assertEquals(10.0, summary.decodeTokS, 1e-9)
    }

    @Test fun resetsThePeakBeforeEveryRun() = runBlocking {
        var resets = 0
        val summary = BenchRunner(ScriptedRuntime(List(3) { done() }), BenchConfig(), pause = {},
            resetPeak = { resets++; true }).run("prompt")
        assertEquals(3, resets)
        assertEquals(true, summary.peakPerRun)
    }

    @Test fun fallsBackToTheWholeProcessPeakWhenTheResetIsRefused() = runBlocking {
        val summary = BenchRunner(ScriptedRuntime(List(3) { done() }), BenchConfig(), pause = {},
            resetPeak = { false }).run("prompt")
        assertEquals(false, summary.peakPerRun)
    }

    @Test fun pausesBeforeEachMeasuredRep() = runBlocking {
        val pauses = mutableListOf<Long>()
        runner(ScriptedRuntime(List(3) { done() }), pauses).run("prompt")
        assertEquals(listOf(5_000L, 5_000L), pauses)
    }

    @Test fun rejectsAReusedPrefix() {
        val runtime = ScriptedRuntime(listOf(done(), done(metrics(reused = 3000))))
        val error = assertThrows(IllegalStateException::class.java) {
            runBlocking { runner(runtime).run("prompt") }
        }
        assertTrue(error.message!!.contains("prefix"))
    }

    @Test fun rejectsAPromptOutsideTheComparableRange() {
        val runtime = ScriptedRuntime(listOf(done(metrics(prompt = 900))))
        val error = assertThrows(IllegalStateException::class.java) {
            runBlocking { runner(runtime).run("prompt") }
        }
        assertTrue(error.message!!.contains("900"))
    }

    @Test fun rejectsADecodeThatStoppedEarly() {
        val runtime = ScriptedRuntime(listOf(done(metrics(generated = 40))))
        val error = assertThrows(IllegalStateException::class.java) {
            runBlocking { runner(runtime).run("prompt") }
        }
        assertTrue(error.message!!.contains("40") && error.message!!.contains("exactly 200"))
    }

    @Test fun rejectsADecodeOneTokenShort() {
        val runtime = ScriptedRuntime(listOf(done(), done(metrics(generated = 199))))
        assertThrows(IllegalStateException::class.java) {
            runBlocking { runner(runtime).run("prompt") }
        }
    }

    @Test fun surfacesEngineFailures() {
        val runtime = ScriptedRuntime(listOf(GenEvent.Failed(Status.IO, "ERR prefill@512")))
        val error = assertThrows(IllegalStateException::class.java) {
            runBlocking { runner(runtime).run("prompt") }
        }
        assertTrue(error.message!!.contains("ERR prefill@512"))
    }

    @Test fun surfacesCancellation() {
        val runtime = ScriptedRuntime(listOf(GenEvent.Cancelled(null)))
        val error = assertThrows(IllegalStateException::class.java) {
            runBlocking { runner(runtime).run("prompt") }
        }
        assertTrue(error.message!!.contains("cancelled"))
    }

    @Test fun rejectsATurnWithoutMetrics() {
        val runtime = ScriptedRuntime(listOf(GenEvent.Done(null)))
        assertThrows(IllegalStateException::class.java) {
            runBlocking { runner(runtime).run("prompt") }
        }
    }

    @Test fun findsTheTierFileInTheFirstRootThatHasIt() {
        val first = tmp.newFolder("first")
        val second = tmp.newFolder("second")
        val model = File(second, "edge0-35b.gguf").apply { writeText("x") }
        assertEquals(model, findModel(listOf(first, null, second), BenchTier.EDGE0_35B))
    }

    @Test fun ignoresOtherTiersAndMissingFiles() {
        val root = tmp.newFolder("root")
        File(root, "edge0-8b.gguf").writeText("x")
        assertNull(findModel(listOf(root), BenchTier.EDGE0_35B))
    }
}
