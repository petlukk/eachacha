#!/usr/bin/env python3
"""Assemble the agent prompt from program.md, kernel, and history."""

import json
import sys
from pathlib import Path


def format_history(entries):
    if not entries:
        return "No previous attempts."
    lines = []
    for e in entries:
        status = "ACCEPTED" if e.get("accepted") else "REJECTED"
        time_str = f"{e['time_us']} µs" if e.get("time_us") else "N/A"
        lines.append(
            f"  #{e['iteration']}: {status} | {time_str} | "
            f"LOC {e.get('loc', '?')} | {e.get('hypothesis', '?')}"
        )
    return "\n".join(lines)


def classify_from_benchmark_output(benchmark_json_str):
    try:
        data = json.loads(benchmark_json_str)
        if not data.get("breakdown"):
            return None
        return _classify_from_breakdown(data["breakdown"])
    except (json.JSONDecodeError, KeyError):
        return None


def _classify_from_breakdown(breakdown):
    DRAM_PEAK_LOW = 25.0
    sizes = list(breakdown.items())
    if not sizes:
        return None
    gbs_values = [(label, vals.get("gbs", 0)) for label, vals in sizes if vals.get("gbs") is not None]
    if not gbs_values:
        return None
    largest_label, largest_gbs = gbs_values[-1]
    lines = []
    if largest_gbs >= DRAM_PEAK_LOW:
        lines.append(f"⚠ BOTTLENECK: DRAM-bound ({largest_gbs:.1f} GB/s at {largest_label})")
        lines.append("  Memory bandwidth is the limit. Only reducing memory traffic helps.")
    elif largest_gbs < 5.0:
        lines.append(f"✓ BOTTLENECK: Compute-bound ({largest_gbs:.1f} GB/s at {largest_label})")
        lines.append("  Significant headroom. Try: wider SIMD, ILP, unrolling, algorithmic restructuring.")
    else:
        lines.append(f"◐ BOTTLENECK: Mixed ({largest_gbs:.1f} GB/s at {largest_label})")
    lines.append("")
    lines.append("  Bandwidth by size:")
    for label, gbs in gbs_values:
        lines.append(f"    {label}: {gbs:.1f} GB/s")
    return "\n".join(lines)


def count_loc(path):
    count = 0
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                count += 1
    return count


def main():
    if len(sys.argv) < 5:
        print("Usage: build_prompt.py <program.md> <kernel.ea> <history.json> "
              "<best_score> [benchmark_json]", file=sys.stderr)
        sys.exit(1)

    program_path = Path(sys.argv[1])
    kernel_path = Path(sys.argv[2])
    history_path = Path(sys.argv[3])
    best_score = sys.argv[4]
    benchmark_json = sys.argv[5] if len(sys.argv) > 5 else None

    program = program_path.read_text()
    kernel = kernel_path.read_text()
    loc = count_loc(kernel_path)

    history = json.loads(history_path.read_text()) if history_path.exists() else []
    last_10 = history[-10:]

    bottleneck_section = ""
    if benchmark_json:
        classification = classify_from_benchmark_output(benchmark_json)
        if classification:
            bottleneck_section = f"\n## Bottleneck Analysis\n{classification}\n"

    prompt = f"""{program}

## Current Best
Score: {best_score} µs (largest dataset size)
LOC: {loc}
{bottleneck_section}
## Current kernel.ea
```ea
{kernel.rstrip()}
```

## History (last {len(last_10)} attempts)
{format_history(last_10)}
"""
    print(prompt)


if __name__ == "__main__":
    main()
