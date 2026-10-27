// BenchTest - on-device benchmark that prints the community submission block.
// Skipped unless `-e tier 8b|35b` is given, so other instrumented runs are unaffected.
// Usage and model staging: android/README.md, "Benchmark".
package dev.edge0.runtime.app

import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import dev.edge0.runtime.app.bench.BenchArgs
import dev.edge0.runtime.app.bench.BenchConfig
import dev.edge0.runtime.app.bench.BenchRunner
import dev.edge0.runtime.app.bench.buildPrompt
import dev.edge0.runtime.app.bench.describeRuns
import dev.edge0.runtime.app.bench.findModel
import dev.edge0.runtime.app.bench.formatSubmissionBlock
import dev.edge0.runtime.app.bench.readDeviceInfo
import dev.edge0.runtime.app.bench.resetPeakMemory
import dev.edge0.runtime.app.runtime.LlamaRuntime
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

class BenchTest {

    @Test
    fun submissionBlock() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val raw = InstrumentationRegistry.getArguments()
        val rawArgs = raw.keySet().associateWith { raw.getString(it) }
        assumeTrue("pass -e ${BenchArgs.KEY_TIER} 8b|35b to run the benchmark",
            BenchArgs.KEY_TIER in rawArgs)
        val args = BenchArgs.parse(rawArgs)
        val context = instrumentation.targetContext

        val model = findModel(modelRoots(context), args.tier)
        assertNotNull("${args.tier.modelFile} not found under files/models (see README, Benchmark)", model)

        val runtime = LlamaRuntime()
        runtime.ensureLoaded(model!!.absolutePath)
        assertTrue("load failed: ${runtime.state.value}", runtime.activeModelDir != null)

        val summary = try {
            BenchRunner(runtime, BenchConfig(temperature = args.temperature), resetPeak = ::resetPeakMemory)
                .run(buildPrompt())
        } finally {
            runtime.unloadActive()
        }

        val shape = summary.reps.first()
        val block = formatSubmissionBlock(
            tier = args.tier,
            device = readDeviceInfo(context),
            summary = summary,
            engineCommit = BuildConfig.ENGINE_COMMIT,
            reporter = args.reporter,
            notes = describeRuns(summary) + listOf(
                "${shape.promptTokens}-token prompt, ${shape.newTokens} decoded, mean of ${summary.reps.size} reps, " +
                    "temperature ${args.temperature}",
                args.notes,
            ),
        )
        publish(context.getExternalFilesDir(null) ?: context.filesDir, args.tier.id, block)
    }

    private fun modelRoots(context: Context): List<File?> = listOf(
        File(context.filesDir, "models"),
        context.getDir("models", 0),
        File(context.cacheDir, "models"),
        context.getExternalFilesDir(null)?.let { File(it, "models") },
    )

    /** Prints through `am instrument` and logcat, then saves a pullable copy; the copy is best effort. */
    private fun publish(outputDir: File, tierId: String, block: String) {
        Log.i(TAG, "\n$block")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.sendStatus(0, Bundle().apply { putString("submission_block", "\n$block") })
        // "stream" is the standard key for a test's console output, which Gradle's connected-test report shows
        instrumentation.sendStatus(0, Bundle().apply { putString("stream", "\n$block\n") })
        runCatching { File(outputDir, "bench-$tierId.txt").writeText(block + "\n") }
            .onFailure { Log.w(TAG, "could not save the block to $outputDir", it) }
    }

    private companion object { const val TAG = "E0Bench" }
}
