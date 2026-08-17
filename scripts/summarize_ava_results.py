#!/usr/bin/env python3
"""Collect AVA-Speech benchmark artifacts into one comparison table.

Reads every report written by ``scripts/benchmark_ava_speech.py`` and prints a
ranked Markdown table. Published reference points are shown alongside, separated
by protocol, because causal and non-causal AVA numbers are not comparable and
mixing them is the most common way these results get misread.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITIONS = ("CLEAN_SPEECH", "SPEECH_WITH_MUSIC", "SPEECH_WITH_NOISE")

# From kiloVAD (arXiv:2607.25870) Table 1; see docs/RESEARCH_NOTES.md.
PUBLISHED_CAUSAL = [
    ("ResectNet", "4.5 k", 0.886),
    ("kiloVAD (full, 360 ms ctx)", "81.1 k", 0.872),
    ("AtomicVAD", "0.3 k", 0.869),
    ("kiloVAD (full, 200 ms ctx)", "81.1 k", 0.862),
    ("MarbleNet", "91 k", 0.850),
    ("kiloVAD (pruned)", "2.1 k", 0.850),
]
PUBLISHED_NON_CAUSAL = [
    ("SincQDR-VAD", "8.0 k", 0.914),
    ("TinyVAD", "11.6 k", 0.864),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="benchmarks/ava-speech", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        overall = report.get("overall", {})
        if "roc_auc" not in overall:
            continue
        interval = report.get("overall_roc_auc_ci95", {})
        adapter = report.get("adapter", {})
        rows.append(
            {
                "model": report["model"],
                "causal": bool(adapter.get("causal", True)),
                "params": adapter.get("parameters"),
                "auc": overall["roc_auc"],
                "lower": interval.get("lower"),
                "upper": interval.get("upper"),
                "f1": overall.get("f1"),
                "fa": overall.get("false_alarm_rate"),
                "miss": overall.get("miss_rate"),
                "conditions": [
                    report["per_condition"].get(name, {}).get("roc_auc")
                    for name in CONDITIONS
                ],
                "hours": report.get("dataset", {}).get("audio_hours"),
            }
        )
    rows.sort(key=lambda row: -row["auc"])

    hours = rows[0]["hours"] if rows else "?"
    print(f"### Measured on this harness ({hours} h of AVA-Speech)\n")
    print("| Model | Params | Causal | AUC | 95% CI | F1 | FA | Miss | clean | music | noise |")
    print("|---|---:|:--:|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        params = f"{row['params']:,}" if row["params"] else "—"
        interval = (
            f"[{row['lower']:.4f}, {row['upper']:.4f}]"
            if row["lower"] is not None
            else "—"
        )
        conditions = " | ".join(
            f"{value:.3f}" if value is not None else "—" for value in row["conditions"]
        )
        print(
            f"| {row['model']} | {params} | {'yes' if row['causal'] else 'no'} | "
            f"**{row['auc']:.4f}** | {interval} | {row['f1']:.4f} | "
            f"{row['fa']:.4f} | {row['miss']:.4f} | {conditions} |"
        )

    print("\n### Published reference points (not measured here)\n")
    print("| Model | Params | AUC | Protocol |")
    print("|---|---:|---:|---|")
    for name, params, auc in PUBLISHED_CAUSAL:
        print(f"| {name} | {params} | {auc:.3f} | strictly causal |")
    for name, params, auc in PUBLISHED_NON_CAUSAL:
        print(f"| {name} | {params} | {auc:.3f} | non-causal, 87.5% overlap |")


if __name__ == "__main__":
    main()
