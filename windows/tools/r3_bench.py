#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r3_bench.py — throughput measurement on the resident tier (48 GB, warm cache).

Each run: 3 reps — long warmup case (a 512-token warmup rep is discarded first) →
256-token greedy decode; parse prompt/eval tok/s from llama-completion; 5s idle
between reps. Absolute values are only meaningful within the same machine and
window when compared against the reference — cross-window comparison is always
suspect (measure on the same day / same process, or treat numbers as shape-only).

Community submission mode (Hardware Benchmarks discussion #142, issue #139):
`--submission` discards the same short warmup, then runs `reps` timed reps
(default 2) on the ~3k-token long-context prompt, averages the numbers, and
prints the ready-to-paste submission block; `--json` prints the same data as
JSON for aggregation.

Usage: python tools/r3_bench.py --tier 35b [--reps 3]
       python tools/r3_bench.py --tier 8b --submission --reporter @you
"""
import argparse, ctypes, json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_bin_dir():
    """Engine dir — same contract as the app (paths.rs::bin_dir): EDGE0_BIN_DIR override,
    else monorepo depot build <repo>/../wt/win/..., else bootstrap <repo>/.edge0/wt/win/..."""
    env = os.environ.get("EDGE0_BIN_DIR")
    if env:
        return env
    for c in (os.path.join(ROOT, "..", "wt", "win", "build-vk", "bin", "Release"),
              os.path.join(ROOT, ".edge0", "wt", "win", "build-vk", "bin", "Release")):
        if os.path.isfile(os.path.join(c, "llama-completion.exe")):
            return c
    return os.path.join(ROOT, "..", "wt", "win", "build-vk", "bin", "Release")


SERVER_BIN = resolve_bin_dir()


def models_root(tier):
    """Where the GGUF lives — repo-local models/ wins when present (dev layout), else the
    app's model home (EDGE0_HOME, default ~/.edge0) where the download+convert step lands it."""
    home = os.environ.get("EDGE0_HOME") or os.path.join(os.path.expanduser("~"), ".edge0")
    roots = [os.path.join(ROOT, "models"), os.path.join(home, "models")]
    for r in roots:
        if os.path.isfile(os.path.join(r, f"edge0-{tier}-gguf", f"edge0-{tier}.gguf")):
            return r
    raise SystemExit(f"no edge0-{tier}.gguf under {roots[0]} or {roots[1]} "
                     "— load/convert once in the app, or run tools/convert_mlx_to_gguf.py first")


PROMPT = ("混合专家模型（MoE）通过稀疏激活把多个专家网络与门控路由组合起来：" * 12)

# Long-context prompt for the submission block (~3k tokens after tokenization):
# identical construction and text to the Python bench's BENCH_LONG (see
# python/examples/bench.py), so prefill numbers on both runtimes refer to the
# same measured work. Short prompts only measure fixed overhead.
_PARA_ZH = '\u6d41\u5f0f\u63a8\u7406\u6846\u67b6\u7684\u8bbe\u8ba1\u8981\u70b9\u5728\u4e8e\u628a\u78c1\u76d8\u5e26\u5bbd\u3001\u7f13\u5b58\u5c42\u7ea7\u4e0e\u8ba1\u7b97\u6838\u5fc3\u4e09\u8005\u91cd\u53e0\u8d77\u6765\u3002\u6df7\u5408\u4e13\u5bb6\u6a21\u578b\u6bcf\u4e00\u5c42\u53ea\u5728\u5c11\u6570\u4e13\u5bb6\u4e0a\u6fc0\u6d3b\uff0c\u8def\u7531\u5668\u7ed9\u51fa\u7684\u7a00\u758f\u9009\u62e9\u6070\u597d\u4e3a\u6309\u9700\u52a0\u8f7d\u63d0\u4f9b\u4e86\u5929\u7136\u7684\u6279\u7c92\u5ea6\u3002\u82e5\u80fd\u5728\u524d\u4e00\u6b65\u7684\u9690\u72b6\u6001\u57fa\u7840\u4e0a\u9884\u5224\u4e0b\u4e00\u6b65\u7684\u4e13\u5bb6\u96c6\u5408\uff0c\u56fa\u6001\u786c\u76d8\u7684\u8bfb\u53d6\u5ef6\u8fdf\u5c31\u80fd\u88ab\u8ba1\u7b97\u5b8c\u5168\u63a9\u76d6\uff0c\u8fd9\u5bf9\u8fb9\u7f18\u8bbe\u5907\u4e0a\u7684\u5927\u6a21\u578b\u90e8\u7f72\u5177\u6709\u76f4\u63a5\u610f\u4e49\u3002'
_PARA_EN = 'The design of a streaming inference framework hinges on overlapping disk bandwidth, cache hierarchy, and compute. A mixture-of-experts layer activates only a handful of experts per token, and the router sparsity provides a natural granularity for on-demand loading. If the expert set of the next step can be predicted from the previous hidden state, SSD read latency hides behind compute entirely, which matters for large-model deployment on edge hardware.'
_LONG_HEADER = '\u8bf7\u4ed4\u7ec6\u9605\u8bfb\u4ee5\u4e0b\u6750\u6599\u5e76\u7528\u4e00\u53e5\u8bdd\u603b\u7ed3\u5176\u6838\u5fc3\u601d\u60f3\uff1a'


def long_prompt():
    paras = [_PARA_ZH if i % 2 == 0 else _PARA_EN for i in range(40)]
    return _LONG_HEADER + " ".join(paras)


PAT = {
    "prompt": re.compile(r"prompt eval time =.*?,\s*([\d.]+) tokens per second"),
    "eval": re.compile(r"(?<!prompt )eval time =.*?,\s*([\d.]+) tokens per second"),
    "total_ms": re.compile(r"total time =\s*([\d.]+) ms"),
    "load_ms": re.compile(r"load time =\s*([\d.]+) ms"),
    "prompt_ms": re.compile(r"prompt eval time =\s*([\d.]+) ms"),
}


# ---------------------------------------------------------------- metadata --
def cpu_name():
    """CPU marketing name from the registry (no wmic/sysinfo dependency)."""
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        name, _ = winreg.QueryValueEx(k, "ProcessorNameString")
        winreg.CloseKey(k)
        return " ".join(str(name).split())
    except OSError:
        return None


def total_ram_gb():
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint),
                        ("dwMemoryLoad", ctypes.c_uint),
                        ("ullTotalPhys", ctypes.c_uint64),
                        ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return st.ullTotalPhys / (1024 ** 3)
    except (OSError, AttributeError):
        return None


def windows_build():
    v = sys.getwindowsversion()
    label = "11" if v.build >= 22000 else "10"
    return f"Windows {label} (build {v.build})"


def machine_name():
    """Computer name (device) from the registry; a human brand is not exposed by winreg."""
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"SYSTEM\CurrentControlSet\Control\ComputerName\ComputerName")
        name, _ = winreg.QueryValueEx(k, "ComputerName")
        winreg.CloseKey(k)
        return str(name)
    except OSError:
        return None


def engine_commit():
    """Head of vendor.llama.pin at the repo root (the pinned llama.cpp fork)."""
    pin = os.path.join(ROOT, "..", "vendor.llama.pin")
    try:
        with open(pin, encoding="utf-8") as f:
            return f.read().splitlines()[0].strip()
    except OSError:
        return None


def vulkan_device(err):
    """Vulkan device name from the engine's own startup log, e.g:
    'ggml_vulkan: 0 = AMD Radeon RX 9070 GRE (vendor) | uma: ...'."""
    for line in err.splitlines():
        if not re.search(r"vulkan", line, re.I):
            continue
        m = re.search(r"\d+\s*=\s*([^\n|()]+)", line)
        if m:
            return m.group(1).strip()
        m = re.search(r"Device\s*\d*\s*[:=]\s*([^\n|(),]+)", line, re.I)
        if m:
            return m.group(1).strip()
    return None


def open_handle(pid):
    """Engine-process handle for the memory probe (Windows only; None otherwise)."""
    if sys.platform != "win32":
        return None
    for _ in range(10):                      # the child may still be spawning
        h = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid)  # QUERY_INFORMATION|VM_READ
        if h:
            return h
        time.sleep(0.01)
    return None


def peak_working_set_gib(handle):
    """Peak process working set (GiB) — the OS keeps the high-water mark while
    any handle to the process is open, so probe after communicate(), then close."""
    if not handle:
        return None
    try:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint),
                        ("PageFaultCount", ctypes.c_uint),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        if not ctypes.WinDLL("psapi").GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return None
        return pmc.PeakWorkingSetSize / (1024 ** 3)
    except OSError:
        return None


def one(gguf, lora, n, tier, prompt=None):
    cmd = [os.path.join(SERVER_BIN, "llama-completion.exe"), "-m", gguf,
           "-ngl", "99", "-cmoe", "--temp", "0", "-n", str(n), "--seed", "7",
           "-no-cnv", "--no-display-prompt", "-p", prompt or PROMPT]
    if lora:
        cmd[3:3] = ["--lora", lora]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, encoding="utf8", errors="replace")
    h = open_handle(p.pid)
    _, err = p.communicate()
    peak = peak_working_set_gib(h)           # probed after wait while the handle is live
    if h:
        ctypes.windll.kernel32.CloseHandle(h)
    out = {"wall_s": round(time.time() - t0, 1),
           "peak_working_set_gib": round(peak, 2) if peak is not None else None,
           "gpu": vulkan_device(err or "")}
    for k, pat in PAT.items():
        m = pat.search(err or "")
        if m:
            out[k] = float(m.group(1))
    ev = re.search(r"n_ctx = (\d+)", err or "")
    out["n_ctx"] = int(ev.group(1)) if ev else None
    if "error" in (err or "").lower() and "eval time" not in err:
        out["error"] = [l for l in err.splitlines() if "error" in l.lower()][:2]
    if "total_ms" in out and "prompt_ms" in out:
        dec_ms = out["total_ms"] - out["prompt_ms"]        # total = generation end-to-end, excludes load (verified)
        out["decode_tps_derived"] = round(n * 1000 / dec_ms, 2) if dec_ms > 0 else None
    return out


def submission_block(tier, reporter, notes, rows):
    """Ready-to-paste Hardware Benchmarks block; numbers are the mean of the timed reps."""
    evs = [r["eval"] for r in rows if "eval" in r]
    prs = [r["prompt"] for r in rows if "prompt" in r]
    peaks = [r["peak_working_set_gib"] for r in rows if r.get("peak_working_set_gib") is not None]
    gpus = sorted({r["gpu"] for r in rows if r.get("gpu")})
    dev, chip = machine_name(), cpu_name()
    ram = total_ram_gb()
    data = {
        "tier": f"edge0-{tier}",
        "runtime": "windows",
        "device": dev or "(unknown)",
        "chip": chip or "(unknown)",
        "gpu": gpus[0] if gpus else "(not detected)",
        "ram_gb": round(ram) if ram else None,
        "os": windows_build(),
        "decode_tok_s": round(sum(evs) / len(evs), 2) if evs else None,
        "prefill_tok_s": round(sum(prs) / len(prs), 2) if prs else None,
        "peak_active_gib": round(max(peaks), 2) if peaks else None,
        "engine_commit": engine_commit(),
        "reporter": reporter or "",
        "notes": notes or "",
    }
    return data


def print_block(d):
    for k, v in d.items():
        print(f"{k:<16} {'' if v is None else v}".rstrip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["35b", "8b"], required=True)
    ap.add_argument("--reps", type=int, default=None,
                    help="timed reps (default: 2 with --submission, 3 otherwise)")
    ap.add_argument("--submission", action="store_true",
                    help="print the Hardware Benchmarks submission block (long-context prefill)")
    ap.add_argument("--json", action="store_true",
                    help="with --submission, also print the block as JSON")
    ap.add_argument("--reporter", default="", help="@handle for the submission block")
    ap.add_argument("--notes", default="", help="free-form notes field")
    a = ap.parse_args()
    reps = a.reps or (2 if a.submission else 3)
    exe = os.path.join(SERVER_BIN, "llama-completion.exe")
    if not os.path.isfile(exe):
        raise SystemExit(f"llama-completion.exe not found in {SERVER_BIN} "
                         "— build the engine (scripts/vendor-build.ps1) or set EDGE0_BIN_DIR")
    md = models_root(a.tier)
    gguf = os.path.join(md, f"edge0-{a.tier}-gguf", f"edge0-{a.tier}.gguf")
    lora = os.path.join(md, f"edge0-{a.tier}",
                        f"lora_edge0_{a.tier}-gguf.gguf")
    if a.submission:
        warm = one(gguf, lora, 64, a.tier)          # warmup (page cache + mmap fault-in), discarded
        print(f"warm(discard): {warm}", flush=True)
        longp = long_prompt()
        rows = []
        for i in range(reps):
            time.sleep(5)
            m = one(gguf, lora, 256, a.tier, prompt=longp)
            m["rep"] = i
            rows.append(m)
            print(f"rep{i}: {m}", flush=True)
        data = submission_block(a.tier, a.reporter, a.notes, rows)
        outp = os.path.join(ROOT, "benchmarks", "r3", f"bench_{a.tier}_submission.json")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, "w", encoding="utf8") as f:
            json.dump({**data, "timed_reps": rows}, f, ensure_ascii=False, indent=1)
        print_block(data)
        if a.json:
            print("SUBMIT_JSON " + json.dumps(data, ensure_ascii=False))
        return
    rows = []
    warm = one(gguf, lora, 64, a.tier)          # warmup (page cache + mmap fault-in), discarded
    print(f"warm(discard): {warm}", flush=True)
    for i in range(reps):
        time.sleep(5)
        m = one(gguf, lora, 256, a.tier)
        m["rep"] = i
        rows.append(m)
        print(f"rep{i}: {m}", flush=True)
    ok = [r for r in rows if "eval" in r]
    evs = sorted(r["eval"] for r in ok)
    prs = sorted(r["prompt"] for r in ok if "prompt" in r)
    summ = dict(tier=a.tier, reps=len(ok), decode_tps_med=evs[len(evs) // 2] if evs else None,
                decode_tps_min=evs[0] if evs else None, decode_tps_max=evs[-1] if evs else None,
                prefill_tps_med=prs[len(prs) // 2] if prs else None, rows=rows,
                note="48 GB tier, -cmoe -ngl99 with LoRA; compare against the same-machine reference")
    outp = os.path.join(ROOT, "benchmarks", "r3", f"bench_{a.tier}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(summ, open(outp, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print("BENCH", json.dumps({k: v for k, v in summ.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
