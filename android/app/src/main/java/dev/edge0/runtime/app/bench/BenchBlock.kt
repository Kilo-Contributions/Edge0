// bench/BenchBlock.kt - the community submission block (field names as in the "Hardware
// Benchmarks" Discussion template) and the device metadata it reports; helpers are pure and unit-tested.
package dev.edge0.runtime.app.bench

import java.util.Locale

private const val BLOCK_KEY_WIDTH = 17

fun formatSubmissionBlock(
    tier: BenchTier,
    device: DeviceInfo,
    summary: BenchSummary,
    engineCommit: String,
    reporter: String,
    notes: List<String>,
): String = listOf(
    "tier" to tier.id,
    "runtime" to "android",
    "device" to device.name,
    "soc" to device.soc,
    "ram_gb" to device.ramGb.toString(),
    "os" to device.os,
    "decode_tok_s" to formatNumber("%.2f", summary.decodeTokS),
    "prefill_tok_s" to formatNumber("%.1f", summary.prefillTokS),
    "peak_active_gib" to formatNumber("%.2f", summary.peakActiveGib),
    "engine_commit" to engineCommit,
    "reporter" to reporter,
    "notes" to notes.filter { it.isNotBlank() }.joinToString("; "),
).joinToString("\n") { (key, value) -> ("$key:".padEnd(BLOCK_KEY_WIDTH) + value).trimEnd() }

private fun formatNumber(pattern: String, value: Double) = String.format(Locale.ROOT, pattern, value)

/** One note per run (cold run first), so the block carries every run and not only the mean. */
fun describeRuns(summary: BenchSummary): List<String> {
    fun line(label: String, run: RepResult) = buildString {
        append("$label: decode ${formatNumber("%.2f", run.decodeTokS)} tok/s")
        append(", prefill ${formatNumber("%.1f", run.prefillTokS)} tok/s")
        if (summary.peakPerRun) append(", peak ${formatNumber("%.2f", run.peakGib)} GiB")
    }
    val runs = listOfNotNull(summary.cold?.let { line("cold run", it) }) +
        summary.reps.mapIndexed { index, run -> line("run ${index + 1}", run) }
    return if (summary.peakPerRun) runs else runs + "peak is the whole-process maximum (per-run reset unavailable)"
}

data class DeviceInfo(val name: String, val soc: String, val ramGb: Int, val os: String) {
    companion object {
        private const val BYTES_PER_GIB = 1L shl 30

        // totalMem excludes kernel-reserved memory, so it reads below the marketed size.
        private val MARKETED_RAM_GB = listOf(2, 3, 4, 6, 8, 10, 12, 16, 18, 24, 32)

        private val QUALCOMM_SOC_NAMES = mapOf(
            "SM8350" to "Snapdragon 888",
            "SM8450" to "Snapdragon 8 Gen 1",
            "SM8475" to "Snapdragon 8+ Gen 1",
            "SM8550" to "Snapdragon 8 Gen 2",
            "SM8650" to "Snapdragon 8 Gen 3",
            "SM8635" to "Snapdragon 8s Gen 3",
            "SM8735" to "Snapdragon 8s Gen 4",
            "SM8750" to "Snapdragon 8 Elite",
            "SM8850" to "Snapdragon 8 Elite Gen 5",
        )

        fun marketedRamGb(totalBytes: Long): Int {
            val gib = totalBytes.toDouble() / BYTES_PER_GIB
            return MARKETED_RAM_GB.firstOrNull { it >= gib } ?: Math.ceil(gib).toInt()
        }

        fun socName(manufacturer: String, model: String): String =
            QUALCOMM_SOC_NAMES[model.trim().uppercase(Locale.ROOT)]
                ?: listOf(manufacturer, model).map { it.trim() }.filter { it.isNotEmpty() }
                    .joinToString(" ").ifEmpty { "unknown" }

        fun deviceName(manufacturer: String, model: String): String {
            val brand = manufacturer.trim().replaceFirstChar { it.uppercase() }
            return if (model.startsWith(brand, ignoreCase = true)) model.trim() else "$brand ${model.trim()}".trim()
        }
    }
}
