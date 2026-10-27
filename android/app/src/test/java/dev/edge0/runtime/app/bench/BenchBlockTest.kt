// BenchBlockTest - submission block field order and formatting, plus the device metadata helpers.
package dev.edge0.runtime.app.bench

import org.junit.Assert.assertEquals
import org.junit.Test

class BenchBlockTest {

    private val device = DeviceInfo(
        name = "Samsung SM-S911B", soc = "Snapdragon 8 Gen 2", ramGb = 8, os = "Android 15 (API 35)")
    private val summary = BenchSummary(listOf(
        RepResult(3300, 200, prefillTokS = 100.04, decodeTokS = 10.123, peakRssBytes = 3L shl 30)))

    private fun gib(value: Double) = (value * (1L shl 30)).toLong()

    @Test fun rendersTemplateFieldsInOrder() {
        val block = formatSubmissionBlock(
            tier = BenchTier.EDGE0_8B, device = device, summary = summary,
            engineCommit = "7ab4ee7", reporter = "@handle", notes = listOf("thermal NONE->LIGHT", "case on"))
        val expected = """
            tier:            edge0-8b
            runtime:         android
            device:          Samsung SM-S911B
            soc:             Snapdragon 8 Gen 2
            ram_gb:          8
            os:              Android 15 (API 35)
            decode_tok_s:    10.12
            prefill_tok_s:   100.0
            peak_active_gib: 3.00
            engine_commit:   7ab4ee7
            reporter:        @handle
            notes:           thermal NONE->LIGHT; case on
        """.trimIndent()
        assertEquals(expected, block)
    }

    @Test fun emptyOptionalFieldsLeaveNoTrailingSpaces() {
        val block = formatSubmissionBlock(
            tier = BenchTier.EDGE0_35B, device = device, summary = summary,
            engineCommit = "7ab4ee7", reporter = "", notes = emptyList())
        assertEquals(listOf("reporter:", "notes:"), block.lines().takeLast(2))
    }

    @Test fun blankNotesAreDropped() {
        val block = formatSubmissionBlock(
            tier = BenchTier.EDGE0_8B, device = device, summary = summary,
            engineCommit = "x", reporter = "@a", notes = listOf("", "only"))
        assertEquals("notes:           only", block.lines().last())
    }

    private fun run(decode: Double, prefill: Double, peakGib: Long) =
        RepResult(3300, 200, prefillTokS = prefill, decodeTokS = decode, peakRssBytes = peakGib shl 30)

    @Test fun describesEveryRunWithPerRunPeaks() {
        val runs = BenchSummary(
            reps = listOf(run(1.5, 10.0, 3), run(1.25, 9.5, 2)), cold = run(1.0, 8.0, 4), peakPerRun = true)
        assertEquals(listOf(
            "cold run: decode 1.00 tok/s, prefill 8.0 tok/s, peak 4.00 GiB",
            "run 1: decode 1.50 tok/s, prefill 10.0 tok/s, peak 3.00 GiB",
            "run 2: decode 1.25 tok/s, prefill 9.5 tok/s, peak 2.00 GiB",
        ), describeRuns(runs))
    }

    @Test fun omitsPerRunPeaksWhenTheResetWasUnavailable() {
        val runs = BenchSummary(reps = listOf(run(1.5, 10.0, 3)), cold = null, peakPerRun = false)
        assertEquals(listOf(
            "run 1: decode 1.50 tok/s, prefill 10.0 tok/s",
            "peak is the whole-process maximum (per-run reset unavailable)",
        ), describeRuns(runs))
    }

    @Test fun ramRoundsUpToTheMarketedSize() {
        assertEquals(8, DeviceInfo.marketedRamGb(gib(7.4)))
        assertEquals(12, DeviceInfo.marketedRamGb(gib(11.2)))
        assertEquals(16, DeviceInfo.marketedRamGb(gib(14.9)))
        assertEquals(6, DeviceInfo.marketedRamGb(gib(5.6)))
    }

    @Test fun ramBeyondTheTableRoundsUpToWholeGib() {
        assertEquals(33, DeviceInfo.marketedRamGb(gib(32.4)))
    }

    @Test fun knownQualcommCodesMapToMarketingNames() {
        assertEquals("Snapdragon 8 Gen 2", DeviceInfo.socName("Qualcomm", "SM8550"))
        assertEquals("Snapdragon 8 Elite", DeviceInfo.socName("QTI", "sm8750"))
        assertEquals("Snapdragon 8s Gen 3", DeviceInfo.socName("Qualcomm", "SM8635"))
        assertEquals("Snapdragon 8s Gen 4", DeviceInfo.socName("Qualcomm", "SM8735"))
        assertEquals("Snapdragon 8 Elite Gen 5", DeviceInfo.socName("Qualcomm", "SM8850"))
    }

    @Test fun unknownSocFallsBackToVendorAndModel() {
        assertEquals("Google Tensor G3", DeviceInfo.socName("Google", "Tensor G3"))
        assertEquals("unknown", DeviceInfo.socName("", ""))
    }

    @Test fun deviceNameAvoidsRepeatingTheManufacturer() {
        assertEquals("Samsung SM-S911B", DeviceInfo.deviceName("samsung", "SM-S911B"))
        assertEquals("Google Pixel 9", DeviceInfo.deviceName("Google", "Google Pixel 9"))
    }
}
