#!/bin/bash
set -euo pipefail

# --- Configuration ---
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
TIMEOUT="${TIMEOUT:-180}"
THRESHOLD="0.5"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

KERNEL_DIR="$SCRIPT_DIR/kernels/search_v2"
KERNEL="$KERNEL_DIR/kernel.ea"
BEST="$KERNEL_DIR/best_kernel.ea"
HISTORY="$KERNEL_DIR/history.json"
PROGRAM="$KERNEL_DIR/program.md"
BENCH="$KERNEL_DIR/bench_kernel.py"

AGENT_OUTPUT="$SCRIPT_DIR/agent_output.txt"
HYPOTHESIS_FILE="$SCRIPT_DIR/hypothesis.txt"

# Allow nested claude invocations
unset CLAUDECODE 2>/dev/null || true

# --- Setup ---
echo "=== Eä Autoresearch: Search v2 Kernel Optimization ==="
echo "Max iterations: $MAX_ITERATIONS"
echo "Timeout per iteration: ${TIMEOUT}s"
echo "Improvement threshold: ${THRESHOLD}%"
echo ""

# Initialize history
[ -f "$HISTORY" ] || echo "[]" > "$HISTORY"

# If best_kernel.ea exists, use it; otherwise seed from the project kernel
if [ -f "$BEST" ]; then
    cp "$BEST" "$KERNEL"
elif [ -f "$REPO_ROOT/chacha20_search_v2.ea" ]; then
    cp "$REPO_ROOT/chacha20_search_v2.ea" "$KERNEL"
    cp "$KERNEL" "$BEST"
else
    echo "ERROR: No kernel found. Expected $REPO_ROOT/chacha20_search_v2.ea"
    exit 1
fi

# Get baseline score
echo "Running baseline benchmark..."
BASELINE=$(python3 "$BENCH" "$BEST")
BENCH_JSON="$BASELINE"
BEST_SCORE=$(echo "$BASELINE" | python3 -c "import sys,json; print(json.load(sys.stdin)['time_us'])")
BEST_LOC=$(echo "$BASELINE" | python3 -c "import sys,json; print(json.load(sys.stdin)['loc'])")

echo "Baseline: ${BEST_SCORE} µs, ${BEST_LOC} LOC"
echo ""

# --- Main Loop ---
for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo "=== Iteration $i / $MAX_ITERATIONS ==="

    # Build prompt
    PROMPT=$(python3 "$SCRIPT_DIR/build_prompt.py" \
        "$PROGRAM" "$BEST" "$HISTORY" "$BEST_SCORE" "$BENCH_JSON")

    # Agent turn
    if ! timeout "$TIMEOUT" claude -p "$PROMPT" --output-format text \
        > "$AGENT_OUTPUT" 2>/dev/null; then
        echo "  TIMEOUT or agent error"
        python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
            "TIMEOUT" "null" "null" "false" "false"
        continue
    fi

    # Parse agent output
    if ! python3 "$SCRIPT_DIR/parse_agent_output.py" \
        "$AGENT_OUTPUT" "$KERNEL" "$HYPOTHESIS_FILE"; then
        echo "  PARSE ERROR"
        python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
            "PARSE_ERROR" "null" "null" "false" "false"
        continue
    fi
    HYPOTHESIS=$(cat "$HYPOTHESIS_FILE")
    echo "  Hypothesis: $HYPOTHESIS"

    # Benchmark (compile + correctness + timing)
    RESULT=""
    if ! RESULT=$(python3 "$BENCH" "$KERNEL" 2>/dev/null); then
        echo "  CRASHED during benchmark"
        cp "$BEST" "$KERNEL"
        python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
            "$HYPOTHESIS" "null" "null" "false" "false" "$KERNEL"
        continue
    fi

    CORRECT=$(echo "$RESULT" | python3 -c "import sys,json; print(str(json.load(sys.stdin)['correct']).lower())")
    TIME_US=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['time_us'] if d['time_us'] else 'null')")
    LOC=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['loc'] if d['loc'] else 'null')")
    ERROR=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error') or '')")

    if [ "$CORRECT" != "true" ]; then
        echo "  REJECTED (incorrect): $ERROR"
        cp "$BEST" "$KERNEL"
        python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
            "$HYPOTHESIS" "$TIME_US" "$LOC" "false" "false" "$KERNEL"
        continue
    fi

    # Evaluate
    ACCEPTED=$(python3 -c "
t, b, threshold = float('$TIME_US'), float('$BEST_SCORE'), float('$THRESHOLD')
l, bl = int('$LOC'), int('$BEST_LOC')
improvement = (b - t) / b * 100
if improvement >= threshold:
    print('true')
elif abs(improvement) < threshold and l < bl:
    print('true')
else:
    print('false')
")

    if [ "$ACCEPTED" = "true" ]; then
        IMPROVEMENT=$(python3 -c "print(f'{(float(\"$BEST_SCORE\") - float(\"$TIME_US\")) / float(\"$BEST_SCORE\") * 100:.2f}')")
        echo "  ACCEPTED: ${TIME_US} µs (${IMPROVEMENT}% improvement), LOC ${LOC}"
        cp "$KERNEL" "$BEST"
        BEST_SCORE="$TIME_US"
        BEST_LOC="$LOC"
        BENCH_JSON="$RESULT"
    else
        echo "  REJECTED: ${TIME_US} µs (best: ${BEST_SCORE} µs), LOC ${LOC}"
        cp "$BEST" "$KERNEL"
    fi

    python3 "$SCRIPT_DIR/log_result.py" "$HISTORY" "$i" \
        "$HYPOTHESIS" "$TIME_US" "$LOC" "true" "$ACCEPTED" "$KERNEL"
done

echo ""
echo "=== Done ==="
echo "Best: ${BEST_SCORE} µs, ${BEST_LOC} LOC"
echo "History: $HISTORY"
echo "Best kernel: $BEST"

# Copy best kernel back to project root if improved
if [ -f "$BEST" ]; then
    cp "$BEST" "$REPO_ROOT/chacha20_search_v2.ea"
    echo "Copied best kernel to $REPO_ROOT/chacha20_search_v2.ea"
fi
