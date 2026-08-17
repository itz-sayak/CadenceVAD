#!/usr/bin/env python3
"""Evaluate a VAD on AVA-Speech under the published causal protocol.

AVA-Speech labels every frame as ``NO_SPEECH`` or one of three speech conditions
(``CLEAN_SPEECH``, ``SPEECH_WITH_MUSIC``, ``SPEECH_WITH_NOISE``). The literature
reports a frame-level ROC-AUC with all speech conditions pooled against
``NO_SPEECH``, plus per-condition breakdowns that pit one speech condition against
the same non-speech pool.

Conditions change *within* a clip, so the model always streams the clip end to end
and frames are sliced by condition afterwards. Slicing the audio per condition
instead would hand a causal model a fresh zero state at every boundary and quietly
inflate its score.

Prepare the set with ``scripts/prepare_ava_speech.py``. This set is evaluation
only: never train, distil, mine negatives, or tune thresholds on it.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from cadencevad.audio import read_audio
from cadencevad.manifest import frame_labels, load_manifest
from cadencevad.metrics import average_precision, best_f1_threshold, binary_metrics, roc_auc

SPEECH_CONDITIONS = ("CLEAN_SPEECH", "SPEECH_WITH_MUSIC", "SPEECH_WITH_NOISE")
ALL_CONDITIONS = ("NO_SPEECH", *SPEECH_CONDITIONS)
CONDITION_CODES = {label: index for index, label in enumerate(ALL_CONDITIONS)}
UNCOVERED_CODE = -1
SAMPLE_RATE = 16_000


def _subset_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float | None,
) -> dict[str, float | int]:
    """Frame metrics for one condition subset."""
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    if positives == 0 or negatives == 0:
        return {"frames": int(labels.size), "positives": positives, "negatives": negatives}
    chosen = threshold
    if chosen is None:
        chosen, _ = best_f1_threshold(probabilities, labels)
    return {
        "frames": int(labels.size),
        "positives": positives,
        "negatives": negatives,
        "speech_fraction": float(np.mean(labels)),
        "roc_auc": roc_auc(probabilities, labels),
        "pr_auc": average_precision(probabilities, labels),
        "threshold": float(chosen),
        **binary_metrics(probabilities, labels, float(chosen)),
    }


def _auc_from_histograms(positive: np.ndarray, negative: np.ndarray) -> float:
    """Tie-corrected Mann-Whitney AUC from binned score counts.

    Resampling whole clips only ever changes how many times each clip's scores are
    counted, so per-clip histograms can be summed and the AUC read straight off the
    pooled counts. That turns each bootstrap replicate into an O(bins) operation
    instead of re-sorting fourteen million frames.
    """
    total_positive = positive.sum()
    total_negative = negative.sum()
    if total_positive == 0 or total_negative == 0:
        return float("nan")
    negatives_below = np.concatenate(([0.0], np.cumsum(negative)[:-1]))
    wins = np.sum(positive * (negatives_below + 0.5 * negative))
    return float(wins / (total_positive * total_negative))


def _bootstrap_auc(
    per_item: list[tuple[np.ndarray, np.ndarray]],
    iterations: int,
    seed: int,
    bins: int = 8_192,
) -> dict[str, float]:
    """Item-level percentile bootstrap over clip-level resampling."""
    if iterations <= 0 or not per_item:
        return {}
    edges = np.linspace(0.0, 1.0, bins + 1)
    positive_histograms = np.empty((len(per_item), bins), dtype=np.float64)
    negative_histograms = np.empty((len(per_item), bins), dtype=np.float64)
    for index, (probabilities, labels) in enumerate(per_item):
        speech = labels == 1
        positive_histograms[index] = np.histogram(probabilities[speech], bins=edges)[0]
        negative_histograms[index] = np.histogram(probabilities[~speech], bins=edges)[0]

    generator = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        picks = generator.integers(0, len(per_item), size=len(per_item))
        samples[index] = _auc_from_histograms(
            positive_histograms[picks].sum(axis=0),
            negative_histograms[picks].sum(axis=0),
        )
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return {}
    lower, upper = np.percentile(finite, (2.5, 97.5))
    return {
        "lower": float(lower),
        "upper": float(upper),
        "mean": float(np.mean(finite)),
        "iterations": int(finite.size),
        "method": f"item bootstrap, {bins}-bin histogram AUC",
    }


def build_adapter(args: argparse.Namespace):
    """Construct the requested adapter from CLI arguments."""
    import vad_adapters as adapters

    if args.model == "cadencevad-torch":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for cadencevad-torch")
        return adapters.CadenceVadTorchAdapter(args.checkpoint, device=args.device)
    if args.model == "cadencevad-onnx":
        return adapters.CadenceVadOnnxAdapter(args.onnx_model, threads=args.threads)
    if args.model == "silero":
        if not args.silero_model:
            raise SystemExit("--silero-model is required for silero")
        return adapters.SileroAdapter(args.silero_model)
    if args.model == "firered":
        if not (args.firered_model and args.firered_cmvn):
            raise SystemExit("--firered-model and --firered-cmvn are required for firered")
        return adapters.FireRedAdapter(args.firered_model, args.firered_cmvn)
    if args.model == "ten":
        if not args.ten_library:
            raise SystemExit("--ten-library is required for ten")
        return adapters.TenAdapter(args.ten_library)
    if args.model == "webrtc":
        return adapters.WebrtcAdapter(args.webrtc_aggressiveness)
    raise SystemExit(f"unknown model {args.model!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="prepare_ava_speech.py output directory")
    parser.add_argument(
        "--model",
        required=True,
        choices=("cadencevad-torch", "cadencevad-onnx", "silero", "firered", "ten", "webrtc"),
    )
    parser.add_argument("--output", help="write the machine-readable report here")
    parser.add_argument("--label", help="override the reported model label")
    parser.add_argument("--limit", type=int, help="evaluate only the first N clips (smoke test)")
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)

    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--onnx-model")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--silero-model")
    parser.add_argument("--firered-model")
    parser.add_argument("--firered-cmvn")
    parser.add_argument("--ten-library")
    parser.add_argument("--webrtc-aggressiveness", type=int, default=2)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "manifest.jsonl"
    provenance = json.loads((dataset / "PROVENANCE.json").read_text(encoding="utf-8"))
    conditions = np.load(dataset / "conditions.npz")

    items = load_manifest(manifest_path)
    if args.limit:
        items = items[: args.limit]

    adapter = build_adapter(args)
    print(f"model={adapter.name} clips={len(items)}")

    from cadencevad.config import FeatureConfig

    feature_config = FeatureConfig()
    all_probabilities: list[np.ndarray] = []
    all_conditions: list[np.ndarray] = []
    per_clip: list[dict[str, object]] = []
    audio_seconds = 0.0
    compute_seconds = 0.0

    for index, item in enumerate(items, 1):
        video_id = str(item.session_id)
        audio = read_audio(item.audio, SAMPLE_RATE)
        codes = np.asarray(conditions[video_id], dtype=np.int8)

        started = time.perf_counter()
        probabilities = adapter.probabilities(audio)
        compute_seconds += time.perf_counter() - started
        audio_seconds += audio.size / SAMPLE_RATE

        usable = min(probabilities.size, codes.size)
        probabilities = np.asarray(probabilities[:usable], dtype=np.float64)
        codes = codes[:usable]

        # Cross-check the manifest's own speech segments against the condition
        # sidecar. They are produced by the same source, so any disagreement means
        # the prepared dataset is inconsistent and the run must not continue.
        expected = frame_labels(item.segments, usable, feature_config)
        derived = np.isin(codes, [CONDITION_CODES[label] for label in SPEECH_CONDITIONS])
        covered = codes != UNCOVERED_CODE
        mismatch = float(np.mean((expected[covered] >= 0.5) != derived[covered]))
        if mismatch > 1e-3:
            raise SystemExit(
                f"{video_id}: manifest segments disagree with condition sidecar "
                f"on {mismatch:.4%} of covered frames"
            )

        all_probabilities.append(probabilities)
        all_conditions.append(codes)
        clip_labels = derived[covered].astype(np.int64)
        clip_probabilities = probabilities[covered]
        per_clip.append(
            {
                "video_id": video_id,
                "frames": int(usable),
                "speech_fraction": float(np.mean(clip_labels)) if clip_labels.size else None,
                "roc_auc": (
                    roc_auc(clip_probabilities, clip_labels)
                    if 0 < int(np.sum(clip_labels)) < clip_labels.size
                    else None
                ),
            }
        )
        if index % 10 == 0 or index == len(items):
            print(
                f"  {index}/{len(items)} clips  ({audio_seconds / 3600:.2f} h, "
                f"{compute_seconds:.1f}s compute)",
                flush=True,
            )

    probabilities = np.concatenate(all_probabilities)
    codes = np.concatenate(all_conditions)
    covered = codes != UNCOVERED_CODE
    speech_codes = [CONDITION_CODES[label] for label in SPEECH_CONDITIONS]

    overall_mask = covered
    overall_labels = np.isin(codes, speech_codes)[overall_mask].astype(np.int64)
    overall = _subset_report(probabilities[overall_mask], overall_labels, None)
    operating_threshold = float(overall["threshold"])

    per_condition: dict[str, dict[str, float | int]] = {}
    for label in SPEECH_CONDITIONS:
        mask = covered & (
            (codes == CONDITION_CODES[label]) | (codes == CONDITION_CODES["NO_SPEECH"])
        )
        subset_labels = (codes[mask] == CONDITION_CODES[label]).astype(np.int64)
        per_condition[label] = _subset_report(
            probabilities[mask], subset_labels, operating_threshold
        )

    per_item_pairs = [
        (
            item_probabilities[item_codes != UNCOVERED_CODE],
            np.isin(item_codes, speech_codes)[item_codes != UNCOVERED_CODE].astype(np.int64),
        )
        for item_probabilities, item_codes in zip(
            all_probabilities, all_conditions, strict=True
        )
    ]

    report = {
        "schema": "cadencevad-ava-speech-evaluation-v1",
        "evaluated_utc": datetime.now(UTC).isoformat(),
        "model": args.label or adapter.name,
        "adapter": adapter.metadata(),
        "protocol": {
            "benchmark": "AVA-Speech",
            "task": "frame-level speech vs non-speech",
            "hop_ms": 10.0,
            "positive_classes": list(SPEECH_CONDITIONS),
            "negative_class": "NO_SPEECH",
            "streaming": (
                "each clip is processed end to end with a single model state; frames "
                "are sliced by condition only after inference"
            ),
            "threshold_policy": (
                "best-F1 threshold selected on the pooled overall subset and then "
                "reused unchanged for every per-condition subset"
            ),
            "smoothing": "none",
            "uncovered_frames_excluded": int(np.count_nonzero(~covered)),
        },
        "dataset": {
            "path": str(dataset),
            "items": len(items),
            "audio_hours": round(audio_seconds / 3600.0, 3),
            "coverage": provenance.get("coverage"),
            "label_sha256": provenance.get("label_sha256"),
            "audio_mirror_revision": provenance.get("audio_mirror_revision"),
        },
        "overall": overall,
        "per_condition": per_condition,
        "overall_roc_auc_ci95": _bootstrap_auc(
            per_item_pairs, args.bootstrap_iterations, args.bootstrap_seed
        ),
        "throughput": {
            "audio_seconds": round(audio_seconds, 2),
            "compute_seconds": round(compute_seconds, 2),
            "real_time_factor": round(compute_seconds / max(audio_seconds, 1e-9), 6),
            "note": "wall-clock of this harness only; not a latency measurement",
        },
        "per_clip": per_clip,
    }

    summary = {
        "model": report["model"],
        "overall_roc_auc": overall.get("roc_auc"),
        "ci95": report["overall_roc_auc_ci95"].get("lower"),
        "f1": overall.get("f1"),
        "false_alarm_rate": overall.get("false_alarm_rate"),
        "miss_rate": overall.get("miss_rate"),
        **{
            label: per_condition[label].get("roc_auc")
            for label in SPEECH_CONDITIONS
        },
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
