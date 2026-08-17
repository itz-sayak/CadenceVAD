#!/usr/bin/env python3
"""Prepare the AVA-Speech causal VAD evaluation set.

AVA-Speech densely annotates speech activity in 160 fifteen-minute YouTube movie
clips with one non-speech and three speech conditions. It is the only public VAD
benchmark carrying a published *strictly causal* leaderboard, so it is the set we
measure CadenceVAD against.

Audio comes from an Apache-2.0 Hugging Face mirror; labels come from Google's own
CC BY 4.0 release. AVA segments start at t=900 s in the source video, so mirror
second ``t`` is official second ``t + 900``. This script verifies that offset per
clip against the mirror's own annotations and refuses to emit a manifest if any
clip disagrees. Most mirror clips carry the full 900 s segment; the realised hours
and any truncation are recorded in ``PROVENANCE.json`` under ``coverage``.

The emitted set is **evaluation only**. It must never be used for training,
distillation targets, hard-negative mining, or threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import ssl
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import soundfile as sf

MIRROR = "nccratliri/vad-human-ava-speech"
MIRROR_PAGE = f"https://huggingface.co/datasets/{MIRROR}"
MIRROR_API = f"https://huggingface.co/api/datasets/{MIRROR}"
MIRROR_RESOLVE = f"https://huggingface.co/datasets/{MIRROR}/resolve"
LABELS_URL = "https://research.google.com/ava/download/ava_speech_labels_v1.csv"
CANONICAL_PAGE = "https://research.google.com/ava/download.html"
PAPER = "https://arxiv.org/abs/1808.00606"

SAMPLE_RATE = 16_000
HOP_MS = 10.0
# AVA annotates seconds 900-1800 of each source video; the mirror ships 900-1200.
AVA_SEGMENT_START_SECONDS = 900.0
SYSTEM_CA_BUNDLE = Path("/etc/ssl/cert.pem")

NON_SPEECH_LABEL = "NO_SPEECH"
SPEECH_LABELS = ("CLEAN_SPEECH", "SPEECH_WITH_MUSIC", "SPEECH_WITH_NOISE")
ALL_LABELS = (NON_SPEECH_LABEL, *SPEECH_LABELS)
# Integer codes written to the per-frame condition sidecar. -1 marks a frame the
# official annotation does not cover, which must be excluded from every metric.
CONDITION_CODES = {label: index for index, label in enumerate(ALL_LABELS)}
UNCOVERED_CODE = -1


def _request_bytes(url: str, attempts: int = 6) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "cadencevad-data/0.1"})
    context = ssl.create_default_context(
        cafile=str(SYSTEM_CA_BUNDLE) if SYSTEM_CA_BUNDLE.exists() else None
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120, context=context) as response:
                return response.read()
        except HTTPError as exc:
            if attempt + 1 == attempts:
                raise
            retry_after = float(exc.headers.get("Retry-After", 0) or 0)
            time.sleep(min(45.0, max(retry_after, 2.0**attempt)))
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(45.0, 2.0**attempt))
    raise AssertionError("unreachable")


def load_official_labels(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Read ava_speech_labels_v1.csv into per-video sorted segment lists."""
    segments: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(f"{path}:{line_number}: expected 4 columns, got {len(row)}")
            video_id, start, end, label = row
            if label not in CONDITION_CODES:
                raise ValueError(f"{path}:{line_number}: unknown label {label!r}")
            segments[video_id].append((float(start), float(end), label))
    for video_id in segments:
        segments[video_id].sort()
    if not segments:
        raise ValueError(f"no segments parsed from {path}")
    return dict(segments)


def condition_codes_for_clip(
    segments: list[tuple[float, float, str]],
    num_frames: int,
    offset_seconds: float,
) -> np.ndarray:
    """Per-frame AVA condition codes for one clip.

    A frame is attributed to the segment covering its end time, matching
    ``cadencevad.manifest.frame_labels`` so probabilities and conditions align
    frame-for-frame.
    """
    frame_ends = (np.arange(num_frames) + 1) * (HOP_MS / 1_000.0) + offset_seconds
    codes = np.full(num_frames, UNCOVERED_CODE, dtype=np.int8)
    for start, end, label in segments:
        codes[(frame_ends > start) & (frame_ends <= end)] = CONDITION_CODES[label]
    return codes


def speech_segments_for_clip(
    segments: list[tuple[float, float, str]],
    offset_seconds: float,
    duration_seconds: float,
) -> list[dict[str, float | str]]:
    """Speech segments rebased onto the clip timeline and merged where adjacent."""
    speech = [
        (max(start - offset_seconds, 0.0), min(end - offset_seconds, duration_seconds))
        for start, end, label in segments
        if label != NON_SPEECH_LABEL
    ]
    speech = [(start, end) for start, end in speech if end > start]
    speech.sort()
    merged: list[list[float]] = []
    for start, end in speech:
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [
        {"start": round(start, 4), "end": round(end, 4), "label": "speech"}
        for start, end in merged
    ]


def _mirror_files(revision: str | None) -> tuple[str, list[str]]:
    metadata = json.loads(_request_bytes(MIRROR_API).decode("utf-8"))
    resolved = revision or str(metadata["sha"])
    names = [str(item["rfilename"]) for item in metadata["siblings"]]
    wavs = sorted(name for name in names if name.endswith("_clip.wav"))
    if not wavs:
        raise ValueError(f"no clip audio found in {MIRROR_PAGE}")
    return resolved, wavs


def _video_id(mirror_name: str) -> str:
    """``train/human_<videoid>_clip.wav`` -> ``<videoid>``."""
    stem = Path(mirror_name).name
    if not stem.startswith("human_") or not stem.endswith("_clip.wav"):
        raise ValueError(f"unexpected mirror filename {mirror_name!r}")
    return stem[len("human_") : -len("_clip.wav")]


def _mirror_speech_mask(annotation: dict[str, object], num_frames: int) -> np.ndarray:
    """Binary speech mask from the mirror's own onset/offset annotation."""
    mask = np.zeros(num_frames, dtype=bool)
    onsets = [float(value) for value in annotation["onset"]]  # type: ignore[index]
    offsets = [float(value) for value in annotation["offset"]]  # type: ignore[index]
    for onset, offset in zip(onsets, offsets, strict=True):
        start = max(int(round(onset / (HOP_MS / 1_000.0))), 0)
        end = min(int(round(offset / (HOP_MS / 1_000.0))), num_frames)
        if end > start:
            mask[start:end] = True
    return mask


def _official_speech_mask(
    segments: list[tuple[float, float, str]],
    num_frames: int,
    offset_seconds: float,
) -> np.ndarray:
    codes = condition_codes_for_clip(segments, num_frames, offset_seconds)
    return np.isin(codes, [CONDITION_CODES[label] for label in SPEECH_LABELS])


def _download_clip(
    revision: str,
    mirror_name: str,
    audio_dir: Path,
) -> tuple[str, Path, bytes, dict[str, object]]:
    video_id = _video_id(mirror_name)
    destination = audio_dir / f"{video_id}.wav"
    annotation_name = mirror_name[: -len(".wav")] + ".json"
    annotation = json.loads(
        _request_bytes(f"{MIRROR_RESOLVE}/{revision}/{annotation_name}").decode("utf-8")
    )
    if destination.exists():
        return video_id, destination, destination.read_bytes(), annotation
    payload = _request_bytes(f"{MIRROR_RESOLVE}/{revision}/{mirror_name}")
    destination.write_bytes(payload)
    return video_id, destination, payload, annotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="destination directory")
    parser.add_argument("--revision", help="pin the mirror revision (default: current sha)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--min-offset-agreement",
        type=float,
        default=0.95,
        help="minimum frame agreement between mirror and official labels per clip",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="write manifest audio paths relative to the manifest",
    )
    args = parser.parse_args()

    output = Path(args.output).resolve()
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    labels_path = output / "ava_speech_labels_v1.csv"
    if not labels_path.exists():
        labels_path.write_bytes(_request_bytes(LABELS_URL))
    labels_digest = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    official = load_official_labels(labels_path)
    segment_total = sum(len(value) for value in official.values())
    print(f"official labels: {segment_total} segments, {len(official)} videos")

    revision, mirror_wavs = _mirror_files(args.revision)
    print(f"mirror {MIRROR}@{revision[:12]}: {len(mirror_wavs)} clips")

    records: list[dict[str, object]] = []
    agreements: list[float] = []
    sources: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_clip, revision, name, audio_dir): name
            for name in mirror_wavs
        }
        for completed, future in enumerate(as_completed(futures), 1):
            video_id, path, payload, annotation = future.result()
            if video_id not in official:
                raise ValueError(f"{video_id} is absent from the official AVA-Speech labels")

            audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
            if sample_rate != SAMPLE_RATE:
                raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sample_rate}")
            if audio.ndim != 1:
                raise ValueError(f"{path}: expected mono audio, got shape {audio.shape}")
            duration = audio.size / SAMPLE_RATE
            num_frames = int(np.ceil(audio.size / (SAMPLE_RATE * HOP_MS / 1_000)))

            segments = official[video_id]
            agreement = float(
                np.mean(
                    _official_speech_mask(segments, num_frames, AVA_SEGMENT_START_SECONDS)
                    == _mirror_speech_mask(annotation, num_frames)
                )
            )
            if agreement < args.min_offset_agreement:
                raise ValueError(
                    f"{video_id}: mirror audio does not align to official AVA labels at "
                    f"offset {AVA_SEGMENT_START_SECONDS} s (frame agreement {agreement:.4f} < "
                    f"{args.min_offset_agreement}). Refusing to emit a misaligned benchmark."
                )
            agreements.append(agreement)

            codes = condition_codes_for_clip(segments, num_frames, AVA_SEGMENT_START_SECONDS)
            audio_reference = (
                str(Path("audio") / path.name) if args.relative_paths else str(path)
            )
            records.append(
                {
                    "video_id": video_id,
                    "manifest": {
                        "audio": audio_reference,
                        "sample_rate": SAMPLE_RATE,
                        "language": "und",
                        "domain": "movie",
                        "channel": "unknown",
                        "codec": "unknown",
                        "device": "unknown",
                        "condition": "ava-speech",
                        "session_id": video_id,
                        "segments": speech_segments_for_clip(
                            segments, AVA_SEGMENT_START_SECONDS, duration
                        ),
                    },
                    "conditions": codes,
                }
            )
            sources.append(
                {
                    "video_id": video_id,
                    "mirror_path": futures[future],
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "duration_seconds": round(duration, 3),
                    "frames": num_frames,
                    "offset_agreement": round(agreement, 6),
                }
            )
            if completed % 20 == 0 or completed == len(futures):
                print(f"  {completed}/{len(futures)} clips verified")

    records.sort(key=lambda record: str(record["video_id"]))
    sources.sort(key=lambda source: str(source["video_id"]))

    manifest_body = "".join(
        json.dumps(record["manifest"], separators=(",", ":")) + "\n" for record in records
    )
    (output / "manifest.jsonl").write_text(manifest_body, encoding="utf-8")
    np.savez_compressed(
        output / "conditions.npz",
        **{str(record["video_id"]): record["conditions"] for record in records},
    )

    total_frames = int(sum(record["conditions"].size for record in records))
    counts = {
        label: int(
            sum(
                np.count_nonzero(record["conditions"] == CONDITION_CODES[label])
                for record in records
            )
        )
        for label in ALL_LABELS
    }
    counts["UNCOVERED"] = int(
        sum(np.count_nonzero(record["conditions"] == UNCOVERED_CODE) for record in records)
    )
    durations = np.array([float(source["duration_seconds"]) for source in sources])

    provenance = {
        "dataset": "AVA-Speech",
        "canonical_page": CANONICAL_PAGE,
        "paper": PAPER,
        "label_license": "CC-BY-4.0 (Google Inc.)",
        "label_url": LABELS_URL,
        "label_sha256": labels_digest,
        "audio_mirror": MIRROR_PAGE,
        "audio_mirror_license": "Apache-2.0",
        "audio_mirror_revision": revision,
        "coverage": {
            "ava_segment_start_seconds": AVA_SEGMENT_START_SECONDS,
            "audio_hours": round(float(np.sum(durations)) / 3600.0, 3),
            "nominal_benchmark_hours": round(len(records) * 900.0 / 3600.0, 3),
            "fraction_of_nominal": round(float(np.sum(durations)) / (len(records) * 900.0), 5),
            "full_length_clips": int(np.count_nonzero(durations >= 890.0)),
            "truncated_clips": int(np.count_nonzero(durations < 890.0)),
            "note": (
                "AVA-Speech annotates seconds 900-1800 of 160 source videos (40 h "
                "nominal). This mirror ships full 900 s audio for most clips and a "
                "300 s excerpt for the rest; see full_length_clips/truncated_clips. "
                "State the realised hours alongside any published comparison."
            ),
        },
        "offset_verification": {
            "method": "frame agreement between mirror annotations and official labels",
            "hop_ms": HOP_MS,
            "min_required": args.min_offset_agreement,
            "min_observed": round(float(np.min(agreements)), 6),
            "mean_observed": round(float(np.mean(agreements)), 6),
        },
        "items": len(records),
        "frames": total_frames,
        "frame_counts": counts,
        "usage": (
            "EVALUATION ONLY. Never train, distil, mine negatives, or tune "
            "thresholds on this set."
        ),
        "sources": sources,
    }
    (output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "items": len(records),
                "frames": total_frames,
                "frame_counts": counts,
                "min_offset_agreement": provenance["offset_verification"]["min_observed"],
                "revision": revision,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
