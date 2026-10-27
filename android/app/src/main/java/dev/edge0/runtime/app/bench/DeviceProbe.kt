// bench/DeviceProbe.kt - reads the device metadata for the submission block and resets the process
// peak-memory counter; both need the Android framework or the Linux /proc interface.
package dev.edge0.runtime.app.bench

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import java.io.File

fun readDeviceInfo(context: Context): DeviceInfo {
    val memory = ActivityManager.MemoryInfo()
    context.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
    return DeviceInfo(
        name = DeviceInfo.deviceName(Build.MANUFACTURER, Build.MODEL),
        soc = DeviceInfo.socName(Build.SOC_MANUFACTURER, Build.SOC_MODEL),
        ramGb = DeviceInfo.marketedRamGb(memory.totalMem),
        os = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})",
    )
}

/** Resets the kernel's peak-RSS counter (VmHWM) to the current RSS; false when the kernel refuses. */
fun resetPeakMemory(): Boolean = runCatching { File("/proc/self/clear_refs").writeText("5") }.isSuccess
