#!/usr/bin/env bash
# run_bench.sh - run the on-device benchmark and print the community submission block.
# Usage: bash tools/bench/run_bench.sh --8b | --35b [--reporter @handle] [--notes "text"] [--temperature 0.2]
#   ADB_SERIAL selects the device; required when more than one is attached.
# Prerequisites: app and test APK installed and the model staged (android/README.md, "Benchmark").
set -euo pipefail

TEST_CLASS=dev.edge0.runtime.app.BenchTest
RUNNER=dev.edge0.runtime.app.test/androidx.test.runner.AndroidJUnitRunner

usage() {
  echo 'usage: run_bench.sh --8b | --35b [--reporter @handle] [--notes "text"] [--temperature 0.2]' >&2
  exit 2
}

# single-quote a value for the device shell
shell_quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

TIER=""
REPORTER=""
NOTES=""
TEMPERATURE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --8b)       TIER=8b ;;
    --35b)      TIER=35b ;;
    --reporter) [ $# -ge 2 ] || usage; REPORTER=$2; shift ;;
    --notes)    [ $# -ge 2 ] || usage; NOTES=$2; shift ;;
    --temperature) [ $# -ge 2 ] || usage; TEMPERATURE=$2; shift ;;
    *)          usage ;;
  esac
  shift
done
[ -n "$TIER" ] || usage

ADB=(adb ${ADB_SERIAL:+-s "$ADB_SERIAL"})
"${ADB[@]}" get-state >/dev/null || {
  echo "ABORT: no usable device (see 'adb devices'; set ADB_SERIAL when several are attached)" >&2
  exit 1
}

command="am instrument -w -e class $TEST_CLASS -e tier $TIER"
if [ -n "$REPORTER" ]; then command="$command -e reporter $(shell_quote "$REPORTER")"; fi
if [ -n "$NOTES" ]; then command="$command -e notes $(shell_quote "$NOTES")"; fi
if [ -n "$TEMPERATURE" ]; then command="$command -e temperature $(shell_quote "$TEMPERATURE")"; fi

echo "Running the $TIER benchmark (a warmup and two timed reps; this can take several minutes)..." >&2
output=$("${ADB[@]}" shell "$command $RUNNER" | tr -d '\r') || true

# the block is a multi-line status value; its start marker follows the test class name on one line
block=$(printf '%s\n' "$output" | sed -n '/INSTRUMENTATION_STATUS: submission_block=/,/^INSTRUMENTATION_STATUS_CODE/{
  /INSTRUMENTATION_STATUS/d
  p
}')

if [ -z "$block" ] || printf '%s\n' "$output" | grep -q 'FAILURES!!!'; then
  echo "ABORT: benchmark did not produce a submission block. Instrumentation output:" >&2
  printf '%s\n' "$output" >&2
  exit 1
fi
printf '%s\n' "$block"
