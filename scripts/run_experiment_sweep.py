#!/usr/bin/env python3
"""Run a sequential training sweep and evaluate every run on AVA-Speech.

Each cell trains one configuration on one manifest with one seed, then measures
it on the held-out benchmark, appending a row to a JSONL results file. Runs are
strictly sequential and use few dataloader workers, so the machine stays usable
across a long sweep.

Results are appended as they complete, so an interrupted sweep loses at most the
run in flight, and re-running skips cells whose results are already present.

AVA-Speech is only ever read here. Nothing in this script selects a model on it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["cell"])
    return done


def write_config(base: Path, overrides: dict, destination: Path) -> None:
    config = json.loads(base.read_text(encoding="utf-8"))
    for section, fields in overrides.items():
        if section == "seed":
            config["seed"] = fields
            continue
        config.setdefault(section, {}).update(fields)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="JSON sweep definition")
    parser.add_argument("--dataset", required=True, type=Path, help="AVA-Speech directory")
    parser.add_argument("--output", required=True, type=Path, help="working directory")
    parser.add_argument("--results", required=True, type=Path, help="JSONL results file")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    base_config = Path(plan["base_config"])
    cells = plan["cells"]
    args.output.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed(args.results)
    print(f"{len(cells)} cells, {len(completed)} already complete", flush=True)

    for index, cell in enumerate(cells, 1):
        name = cell["cell"]
        if name in completed:
            print(f"[{index}/{len(cells)}] skip {name}", flush=True)
            continue

        started = time.time()
        workdir = args.output / name
        config_path = workdir / "config.json"
        write_config(base_config, cell.get("overrides", {}), config_path)

        print(f"[{index}/{len(cells)}] train {name}", flush=True)
        code = run(
            [
                "uv", "run", "--no-sync", "cadencevad", "train",
                "--config", str(config_path),
                "--train-manifest", cell["train_manifest"],
                "--valid-manifest", cell["valid_manifest"],
                "--output", str(workdir),
                "--device", args.device,
            ],
            workdir / "train.log",
        )
        if code != 0:
            print(f"    training failed ({code}); see {workdir / 'train.log'}", flush=True)
            continue

        report = workdir / "ava.json"
        print(f"[{index}/{len(cells)}] evaluate {name}", flush=True)
        code = run(
            [
                "uv", "run", "--no-sync", "python", "scripts/benchmark_ava_speech.py",
                "--dataset", str(args.dataset),
                "--model", "cadencevad-torch",
                "--checkpoint", str(workdir / "best.pt"),
                "--device", args.device,
                "--label", name,
                "--output", str(report),
                "--bootstrap-iterations", "500",
            ],
            workdir / "evaluate.log",
        )
        if code != 0 or not report.exists():
            print(f"    evaluation failed ({code}); see {workdir / 'evaluate.log'}", flush=True)
            continue

        measured = json.loads(report.read_text(encoding="utf-8"))
        overall = measured["overall"]
        calibration = json.loads(
            (workdir / "detector-calibration.json").read_text(encoding="utf-8")
        )["metrics"]
        row = {
            "cell": name,
            "group": cell.get("group", ""),
            "variable": cell.get("variable"),
            "seed": cell.get("overrides", {}).get("seed"),
            "parameters": measured["adapter"].get("parameters"),
            "ava_roc_auc": overall["roc_auc"],
            "ava_ci95": measured.get("overall_roc_auc_ci95", {}),
            "ava_f1": overall["f1"],
            "ava_false_alarm_rate": overall["false_alarm_rate"],
            "ava_miss_rate": overall["miss_rate"],
            "ava_per_condition": {
                key: value.get("roc_auc") for key, value in measured["per_condition"].items()
            },
            "dev_detector_f1": calibration["f1"],
            "dev_false_alarm_rate": calibration["false_alarm_rate"],
            "dev_miss_rate": calibration["miss_rate"],
            "train_manifest": cell["train_manifest"],
            "elapsed_seconds": round(time.time() - started, 1),
        }
        with args.results.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"    AUC {row['ava_roc_auc']:.4f}  FA {row['ava_false_alarm_rate']:.4f}  "
            f"miss {row['ava_miss_rate']:.4f}  ({row['elapsed_seconds']:.0f}s)",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
