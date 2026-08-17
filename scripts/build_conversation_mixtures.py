#!/usr/bin/env python3
"""Synthesise call-shaped training audio with frame-accurate speech labels.

CadenceVAD v0.1 was trained on FLEURS read-speech clips that are roughly 95% speech.
A real call is nothing like that: it alternates speech with pauses, breaths, room
tone, music beds and noise, and it spends a large fraction of its time in silence.
A model trained on the former and deployed on the latter fires constantly, which is
what a 26% false-alarm rate looks like.

This script builds the missing distribution. It lays teacher-verified speech
regions onto a timeline at a controlled duty cycle, fills the gaps with real noise
and room tone, and applies channel effects, writing exact per-frame supervision as
it goes. Labels come from where each region was *placed*, so they are correct by
construction rather than inferred after the fact.

Speakers never cross the train/validation boundary, and each clip records its
``speaker_id`` so ``cadencevad.manifest.validate_manifest_group_separation`` can
prove it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from cadencevad.audio import read_audio, telephone_roundtrip

SAMPLE_RATE = 16_000
HOP_MS = 10.0
HOP_SAMPLES = 160

# Teacher probability above which a source frame counts as speech when trimming a
# source utterance down to its spoken region.
TEACHER_SPEECH_THRESHOLD = 0.5
# Ignore utterances whose teacher-detected speech is shorter than this.
MIN_UTTERANCE_SECONDS = 0.3


@dataclass
class MixtureSpec:
    """Per-clip randomisation ranges."""

    clip_seconds: float = 30.0
    duty_cycle_min: float = 0.30
    duty_cycle_max: float = 0.80
    # Silence between consecutive utterances. Short gaps dominate (within-turn
    # pauses), with a long tail for turn boundaries and hold time.
    gap_seconds_min: float = 0.15
    gap_seconds_max: float = 4.0
    gap_long_probability: float = 0.25
    noise_probability: float = 0.85
    noise_snr_db_min: float = -5.0
    noise_snr_db_max: float = 25.0
    rir_probability: float = 0.35
    telephone_probability: float = 0.40
    gain_db_min: float = -18.0
    gain_db_max: float = 3.0
    clipping_probability: float = 0.10
    # Fraction of clips that contain no speech at all. These are the pure hard
    # negatives that teach the model to stay quiet.
    silent_clip_probability: float = 0.12


@dataclass
class SourceIndex:
    speech: dict[str, list[Path]] = field(default_factory=dict)
    noise: list[Path] = field(default_factory=list)
    rirs: list[Path] = field(default_factory=list)


def _speaker_of(path: Path, root: Path) -> str:
    """LibriSpeech lays files out as ``<split>/<speaker>/<chapter>/<id>.flac``."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.parent.name
    parts = relative.parts
    return parts[1] if len(parts) >= 3 else path.parent.name


def index_sources(
    speech_root: Path,
    noise_roots: list[Path],
    rir_root: Path | None,
    speech_pattern: str,
) -> SourceIndex:
    speech: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(speech_root.rglob(speech_pattern)):
        speech[_speaker_of(path, speech_root)].append(path)
    if not speech:
        raise SystemExit(f"no speech matching {speech_pattern!r} under {speech_root}")

    noise: list[Path] = []
    for root in noise_roots:
        noise.extend(sorted(root.rglob("*.wav")))
        noise.extend(sorted(root.rglob("*.flac")))
    if not noise:
        raise SystemExit(f"no noise found under {[str(r) for r in noise_roots]}")

    rirs: list[Path] = []
    if rir_root is not None:
        # Only the simulated/real room responses; the pointsource *noise* files
        # living in the same archive are noise, not impulse responses.
        rirs = sorted(
            path
            for path in rir_root.rglob("*.wav")
            if "rir" in str(path).lower() and "pointsource" not in str(path).lower()
        )
    return SourceIndex(speech=dict(speech), noise=noise, rirs=rirs)


def _apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with a room response, preserving length and onset alignment."""
    response = np.asarray(rir, dtype=np.float32).reshape(-1)
    if response.size == 0:
        return audio
    peak = int(np.argmax(np.abs(response)))
    response = response[peak : peak + 4_000]
    if response.size == 0 or not np.any(response):
        return audio
    response = response / (np.max(np.abs(response)) + 1e-9)
    convolved = np.convolve(audio, response)[: audio.size]
    return convolved.astype(np.float32)


def _mix_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    speech_mask: np.ndarray,
) -> np.ndarray:
    """Mix noise under a signal at an SNR measured over *speech* frames only.

    Measuring over the whole clip would make the SNR meaningless: a mostly silent
    clip has almost no signal energy, so a nominal 10 dB would in practice bury the
    speech. The reference level therefore comes from the speech regions.
    """
    if noise.size < signal.size:
        noise = np.tile(noise, int(np.ceil(signal.size / max(noise.size, 1))))
    noise = noise[: signal.size]
    reference = signal[speech_mask] if np.any(speech_mask) else signal
    signal_rms = float(np.sqrt(np.mean(reference**2) + 1e-10))
    noise_rms = float(np.sqrt(np.mean(noise**2) + 1e-10))
    if noise_rms <= 1e-9:
        return signal
    scale = signal_rms / (10 ** (snr_db / 20.0) * noise_rms)
    return (signal + noise * scale).astype(np.float32)


def _teacher_speech_span(probabilities: np.ndarray) -> tuple[int, int] | None:
    """First and last frame the teacher calls speech."""
    speech = np.flatnonzero(probabilities >= TEACHER_SPEECH_THRESHOLD)
    if speech.size == 0:
        return None
    return int(speech[0]), int(speech[-1]) + 1


class _Teacher:
    """Lazily constructed FireRedVAD/Silero ensemble, one per worker process."""

    def __init__(self, firered_model: str | None, firered_cmvn: str | None, silero: str | None):
        self.firered = None
        self.silero = None
        if firered_model and firered_cmvn:
            from cadencevad.teacher import FireRedOnnxTeacher

            self.firered = FireRedOnnxTeacher(firered_model, firered_cmvn)
        if silero:
            from cadencevad.teacher import SileroOnnxTeacher

            self.silero = SileroOnnxTeacher(silero)
        if self.firered is None and self.silero is None:
            raise SystemExit("at least one teacher model is required")

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        frames = int(np.ceil(audio.size / HOP_SAMPLES))
        outputs = []
        for teacher in (self.firered, self.silero):
            if teacher is None:
                continue
            values = np.asarray(
                teacher.predict(audio, sample_rate=SAMPLE_RATE, target_hop_ms=HOP_MS),
                dtype=np.float32,
            ).reshape(-1)
            if values.size < frames:
                values = np.pad(values, (0, frames - values.size), mode="edge")
            outputs.append(values[:frames])
        # The mean of an aligned ensemble keeps genuine disagreement as an
        # intermediate probability instead of forcing a hard call, which is
        # exactly the uncertainty the soft-target blend downstream expects.
        return np.clip(np.mean(outputs, axis=0), 0.0, 1.0)


_WORKER: dict[str, object] = {}


def _init_worker(firered_model, firered_cmvn, silero_model, threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    _WORKER["teacher"] = _Teacher(firered_model, firered_cmvn, silero_model)


def _build_clip(job: dict) -> dict | None:
    teacher: _Teacher = _WORKER["teacher"]  # type: ignore[assignment]
    spec: MixtureSpec = job["spec"]
    rng = np.random.default_rng(job["seed"])
    total_samples = int(round(spec.clip_seconds * SAMPLE_RATE))
    frames = total_samples // HOP_SAMPLES

    timeline = np.zeros(total_samples, dtype=np.float32)
    labels = np.zeros(frames, dtype=np.float32)
    speakers: set[str] = set()

    silent = rng.random() < spec.silent_clip_probability
    if not silent:
        duty = float(rng.uniform(spec.duty_cycle_min, spec.duty_cycle_max))
        speech_budget = int(duty * total_samples)
        cursor = int(rng.uniform(0.0, 1.5) * SAMPLE_RATE)
        placed = 0
        attempts = 0
        while placed < speech_budget and cursor < total_samples and attempts < 256:
            attempts += 1
            speaker, source = job["utterances"][rng.integers(0, len(job["utterances"]))]
            try:
                audio = read_audio(Path(source), SAMPLE_RATE)
            except Exception:
                continue
            if audio.size < int(MIN_UTTERANCE_SECONDS * SAMPLE_RATE):
                continue
            probabilities = teacher.probabilities(audio)
            span = _teacher_speech_span(probabilities)
            if span is None:
                continue
            start_frame, end_frame = span
            audio = audio[start_frame * HOP_SAMPLES : end_frame * HOP_SAMPLES]
            probabilities = probabilities[start_frame:end_frame]
            if audio.size < int(MIN_UTTERANCE_SECONDS * SAMPLE_RATE):
                continue

            # A LibriSpeech utterance averages around twelve seconds, which would
            # swamp a thirty-second clip and push the duty cycle far past its
            # target. Take a turn-length window instead: conversational turns are
            # mostly short with a long tail, and the remaining speech budget caps
            # the draw so the realised duty cycle tracks the requested one.
            remaining_frames = max(0, (speech_budget - placed) // HOP_SAMPLES)
            if remaining_frames <= 0:
                break
            turn_seconds = float(
                np.clip(rng.lognormal(mean=np.log(2.0), sigma=0.8), 0.4, 9.0)
            )
            turn_frames = int(turn_seconds * 1_000 / HOP_MS)
            turn_frames = min(turn_frames, remaining_frames, probabilities.size)
            if turn_frames < int(MIN_UTTERANCE_SECONDS * 1_000 / HOP_MS):
                turn_frames = min(
                    probabilities.size, int(MIN_UTTERANCE_SECONDS * 1_000 / HOP_MS)
                )
            offset_in_source = (
                int(rng.integers(0, probabilities.size - turn_frames + 1))
                if probabilities.size > turn_frames
                else 0
            )
            audio = audio[
                offset_in_source * HOP_SAMPLES : (offset_in_source + turn_frames)
                * HOP_SAMPLES
            ]
            probabilities = probabilities[
                offset_in_source : offset_in_source + turn_frames
            ]
            if audio.size == 0:
                continue

            # Align the placement to the frame grid so labels stay exact.
            offset_frame = cursor // HOP_SAMPLES
            offset = offset_frame * HOP_SAMPLES
            usable = min(audio.size, total_samples - offset)
            if usable <= 0:
                break
            usable_frames = usable // HOP_SAMPLES
            if usable_frames <= 0:
                break
            timeline[offset : offset + usable_frames * HOP_SAMPLES] += audio[
                : usable_frames * HOP_SAMPLES
            ]
            labels[offset_frame : offset_frame + usable_frames] = np.maximum(
                labels[offset_frame : offset_frame + usable_frames],
                probabilities[:usable_frames],
            )
            placed += usable_frames * HOP_SAMPLES
            speakers.add(speaker)

            if rng.random() < spec.gap_long_probability:
                gap = rng.uniform(1.0, spec.gap_seconds_max)
            else:
                gap = rng.uniform(spec.gap_seconds_min, 1.0)
            cursor = offset + usable_frames * HOP_SAMPLES + int(gap * SAMPLE_RATE)

    peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
    if peak > 1e-6:
        timeline = timeline / peak * 0.7

    speech_mask = np.repeat(labels >= 0.5, HOP_SAMPLES)[:total_samples]
    if speech_mask.size < total_samples:
        speech_mask = np.pad(speech_mask, (0, total_samples - speech_mask.size))

    if job["rirs"] and rng.random() < spec.rir_probability:
        try:
            rir = read_audio(Path(job["rirs"][rng.integers(0, len(job["rirs"]))]), SAMPLE_RATE)
            timeline = _apply_rir(timeline, rir)
        except Exception:
            pass

    condition = "clean"
    snr_db = None
    if job["noise"] and rng.random() < spec.noise_probability:
        try:
            noise_path = Path(job["noise"][rng.integers(0, len(job["noise"]))])
            noise = read_audio(noise_path, SAMPLE_RATE)
            if noise.size > 0:
                if noise.size > total_samples:
                    start = int(rng.integers(0, noise.size - total_samples + 1))
                    noise = noise[start : start + total_samples]
                snr_db = float(rng.uniform(spec.noise_snr_db_min, spec.noise_snr_db_max))
                timeline = _mix_at_snr(timeline, noise, snr_db, speech_mask)
                condition = noise_path.parent.parent.name or "noise"
        except Exception:
            pass

    timeline = timeline * float(10 ** (rng.uniform(spec.gain_db_min, spec.gain_db_max) / 20.0))

    codec = "pcm16"
    if rng.random() < spec.telephone_probability:
        codec = "g711-ulaw" if rng.random() < 0.5 else "g711-alaw"
        timeline = telephone_roundtrip(
            timeline, SAMPLE_RATE, codec="mu-law" if "ulaw" in codec else "a-law"
        )

    if rng.random() < spec.clipping_probability:
        limit = float(rng.uniform(0.25, 0.9))
        timeline = np.clip(timeline, -limit, limit) / limit

    timeline = np.clip(timeline, -1.0, 1.0).astype(np.float32)
    if not np.all(np.isfinite(timeline)):
        return None

    audio_path = Path(job["audio_path"])
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, timeline, SAMPLE_RATE, subtype="PCM_16")
    teacher_path = audio_path.with_suffix(".teacher.npy")
    np.save(teacher_path, labels.astype(np.float32))

    hard = labels >= 0.5
    segments = []
    if np.any(hard):
        padded = np.pad(hard.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        for start, end in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True
        ):
            segments.append(
                {
                    "start": round(float(start) * HOP_MS / 1_000.0, 4),
                    "end": round(float(end) * HOP_MS / 1_000.0, 4),
                    "label": "speech",
                }
            )

    return {
        "split": job["split"],
        "speech_fraction": float(np.mean(hard)),
        "manifest": {
            "audio": os.path.relpath(audio_path, Path(job["manifest_dir"])),
            "sample_rate": SAMPLE_RATE,
            "language": "en",
            "domain": "synthetic-call",
            "channel": "mixed",
            "codec": codec,
            "device": "unknown",
            "condition": condition,
            "snr_db": snr_db,
            "speaker_id": "+".join(sorted(speakers)) or None,
            "session_id": audio_path.stem,
            "teacher_probabilities": os.path.relpath(teacher_path, Path(job["manifest_dir"])),
            "teacher_weight": 1.0,
            "teacher_confidence_weighting": False,
            "segments": segments,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-root", required=True, type=Path)
    parser.add_argument("--speech-pattern", default="*.flac")
    parser.add_argument("--noise-root", required=True, type=Path, action="append")
    parser.add_argument("--rir-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-hours", type=float, default=40.0)
    parser.add_argument("--valid-hours", type=float, default=4.0)
    parser.add_argument("--clip-seconds", type=float, default=30.0)
    parser.add_argument("--valid-speaker-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument("--firered-model")
    parser.add_argument("--firered-cmvn")
    parser.add_argument("--silero-model")
    parser.add_argument("--noise-snr-db-min", type=float, default=MixtureSpec.noise_snr_db_min)
    parser.add_argument("--noise-snr-db-max", type=float, default=MixtureSpec.noise_snr_db_max)
    parser.add_argument("--noise-probability", type=float, default=MixtureSpec.noise_probability)
    parser.add_argument("--rir-probability", type=float, default=MixtureSpec.rir_probability)
    parser.add_argument("--duty-cycle-min", type=float, default=MixtureSpec.duty_cycle_min)
    parser.add_argument("--duty-cycle-max", type=float, default=MixtureSpec.duty_cycle_max)
    parser.add_argument(
        "--silent-clip-probability",
        type=float,
        default=MixtureSpec.silent_clip_probability,
    )
    args = parser.parse_args()

    spec = MixtureSpec(
        clip_seconds=args.clip_seconds,
        noise_snr_db_min=args.noise_snr_db_min,
        noise_snr_db_max=args.noise_snr_db_max,
        noise_probability=args.noise_probability,
        rir_probability=args.rir_probability,
        duty_cycle_min=args.duty_cycle_min,
        duty_cycle_max=args.duty_cycle_max,
        silent_clip_probability=args.silent_clip_probability,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sources = index_sources(
        args.speech_root.resolve(),
        [root.resolve() for root in args.noise_root],
        args.rir_root.resolve() if args.rir_root else None,
        args.speech_pattern,
    )
    speakers = sorted(sources.speech)
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    split_at = max(1, int(len(speakers) * args.valid_speaker_fraction))
    valid_speakers = set(speakers[:split_at])
    train_speakers = set(speakers[split_at:])
    print(
        f"speech: {sum(len(v) for v in sources.speech.values())} files, "
        f"{len(speakers)} speakers ({len(train_speakers)} train / {len(valid_speakers)} valid); "
        f"noise: {len(sources.noise)} files; rirs: {len(sources.rirs)}"
    )

    utterances = {
        "train": [
            (speaker, str(path))
            for speaker in train_speakers
            for path in sources.speech[speaker]
        ],
        "valid": [
            (speaker, str(path))
            for speaker in valid_speakers
            for path in sources.speech[speaker]
        ],
    }
    noise = [str(path) for path in sources.noise]
    rirs = [str(path) for path in sources.rirs]

    jobs = []
    for split, hours in (("train", args.train_hours), ("valid", args.valid_hours)):
        count = int(round(hours * 3600.0 / spec.clip_seconds))
        for index in range(count):
            jobs.append(
                {
                    "split": split,
                    "seed": args.seed + (0 if split == "train" else 1_000_000) + index,
                    "spec": spec,
                    "utterances": utterances[split],
                    "noise": noise,
                    "rirs": rirs,
                    "audio_path": str(output / split / f"{split}-{index:06d}.wav"),
                    "manifest_dir": str(output),
                }
            )
    print(f"building {len(jobs)} clips ({args.train_hours} h train + {args.valid_hours} h valid)")

    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.firered_model, args.firered_cmvn, args.silero_model, 1),
    ) as pool:
        futures = [pool.submit(_build_clip, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                results.append(result)
            if completed % 100 == 0 or completed == len(futures):
                print(f"  {completed}/{len(futures)} clips", flush=True)

    for split in ("train", "valid"):
        rows = sorted(
            (item for item in results if item["split"] == split),
            key=lambda item: str(item["manifest"]["audio"]),
        )
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row["manifest"], separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        fractions = np.array([row["speech_fraction"] for row in rows])
        print(
            f"{split}: {len(rows)} clips, speech fraction "
            f"mean {fractions.mean():.3f} p10 {np.percentile(fractions, 10):.3f} "
            f"p90 {np.percentile(fractions, 90):.3f}"
        )

    provenance = {
        "generator": "scripts/build_conversation_mixtures.py",
        "seed": args.seed,
        "clip_seconds": spec.clip_seconds,
        "speech_root": str(args.speech_root),
        "noise_roots": [str(root) for root in args.noise_root],
        "rir_root": str(args.rir_root) if args.rir_root else None,
        "teachers": {
            "firered": args.firered_model,
            "silero": args.silero_model,
            "ensemble": "mean of available teachers",
        },
        "speaker_split": {
            "train_speakers": sorted(train_speakers),
            "valid_speakers": sorted(valid_speakers),
        },
        "labels": (
            "Speech regions are placed on the timeline at known frame offsets, so "
            "supervision is exact by construction; within-utterance detail comes "
            "from the teacher ensemble."
        ),
    }
    (output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
