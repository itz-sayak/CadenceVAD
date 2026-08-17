#!/usr/bin/env python3
"""Uniform per-frame probability adapters for CadenceVAD and external VAD baselines.

Every adapter maps 16 kHz mono float32 audio to one speech probability per 10 ms
frame, so accuracy comparisons run on an identical frame grid regardless of each
model's native hop. Models with a coarser native hop (Silero uses 32 ms) are
interpolated with the repository's existing
:func:`cadencevad.teacher.interpolate_probabilities`.

These adapters are for **accuracy** measurement. Latency belongs to
``scripts/benchmark_official_vads.py``, which measures each runtime in its own
native configuration; do not read speed numbers off this module.

External model code and weights are loaded from user-supplied checkouts and are
never vendored into this repository.
"""

from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
HOP_MS = 10.0
HOP_SAMPLES = 160


def target_frames(num_samples: int) -> int:
    """Frame count on the shared 10 ms grid, matching ``manifest.frame_labels``."""
    return int(np.ceil(num_samples / HOP_SAMPLES))


def _fit(probabilities: np.ndarray, frames: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if values.size == frames:
        return values
    if values.size > frames:
        return values[:frames]
    return np.pad(values, (0, frames - values.size), mode="edge")


class VadAdapter(ABC):
    """One speech probability per 10 ms frame."""

    name: str = "unnamed"
    hop_ms: float = HOP_MS
    causal: bool = True
    scope: str = ""

    @abstractmethod
    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        """Return float32 probabilities of length ``target_frames(audio.size)``."""

    def metadata(self) -> dict[str, object]:
        return {
            "adapter": self.name,
            "native_hop_ms": self.hop_ms,
            "causal": self.causal,
            "scope": self.scope,
        }


class CadenceVadTorchAdapter(VadAdapter):
    """CadenceVAD checkpoint evaluated as a full causal sequence.

    The model pads every depthwise convolution on the left only and starts the GRU
    from a zero state, so a whole-sequence forward pass is numerically the same
    computation as the 10 ms streaming loop, just batched. ``tests/test_streaming.py``
    pins that equivalence. Running it offline lets a 900 s clip go through the GPU
    in one call instead of 90,000 Python steps.
    """

    causal = True
    scope = "causal frontend + model, offline-batched streaming-equivalent"

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        name: str | None = None,
        recurrent_chunk_frames: int = 8_192,
    ):
        import torch

        from cadencevad.benchmark import load_checkpoint
        from cadencevad.features import CausalFeatureExtractor

        self._torch = torch
        self.config, model = load_checkpoint(checkpoint)
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.frontend = CausalFeatureExtractor(self.config.feature).to(self.device).eval()
        self.hop_ms = self.config.feature.hop_ms
        self.name = name or f"cadencevad-torch:{Path(checkpoint).stem}"
        self.parameter_count = self.model.parameter_count
        self.recurrent_chunk_frames = int(recurrent_chunk_frames)

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        torch = self._torch
        frames = target_frames(audio.size)
        model = self.model
        with torch.inference_mode():
            waveform = torch.from_numpy(np.ascontiguousarray(audio)).to(self.device)
            features = self.frontend(waveform.unsqueeze(0))

            # The convolution stack handles the whole clip at once, but cuDNN's GRU
            # rejects sequences this long. Feeding the GRU in chunks while carrying
            # its hidden state forward is the same computation, just split up.
            encoded = model.encode(features)
            hidden = torch.zeros(
                1,
                encoded.shape[0],
                model.config.recurrent_dim,
                device=encoded.device,
                dtype=encoded.dtype,
            )
            outputs = []
            for start in range(0, encoded.shape[1], self.recurrent_chunk_frames):
                chunk = encoded[:, start : start + self.recurrent_chunk_frames].contiguous()
                output, hidden = model.recurrent(chunk, hidden)
                outputs.append(output)
            encoded = torch.cat(outputs, dim=1)
            logits = model._heads(encoded)["speech_logits"]
            values = torch.sigmoid(logits).squeeze(0).float().cpu().numpy()
        return _fit(values, frames)

    def metadata(self) -> dict[str, object]:
        return {**super().metadata(), "parameters": self.parameter_count}


class CadenceVadOnnxAdapter(VadAdapter):
    """CadenceVAD through the shipped streaming ONNX runtime, one 10 ms hop at a time."""

    causal = True
    scope = "causal frontend + model via OnnxStreamingVadModel"

    def __init__(self, model_path: str | Path | None = None, threads: int = 1):
        from cadencevad.runtime import OnnxStreamingVadModel

        if model_path is None:
            self.owner = OnnxStreamingVadModel.load_bundled(threads=threads)
            self.name = "cadencevad-onnx:bundled"
        else:
            self.owner = OnnxStreamingVadModel(model_path, threads=threads)
            self.name = f"cadencevad-onnx:{Path(model_path).stem}"

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        stream = self.owner.new_stream()
        values, _ = stream.push(np.ascontiguousarray(audio, dtype=np.float32))
        return _fit(np.asarray(values, dtype=np.float32), target_frames(audio.size))


class SileroAdapter(VadAdapter):
    """Official Silero ONNX detector, resampled from its native 32 ms hop."""

    hop_ms = 32.0
    causal = True
    scope = "official model + recurrent state, interpolated to the 10 ms grid"

    def __init__(self, model_path: str | Path):
        from cadencevad.teacher import SileroOnnxTeacher

        self.teacher = SileroOnnxTeacher(model_path)
        self.name = "silero-vad"

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        values = self.teacher.predict(
            np.ascontiguousarray(audio, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
            target_hop_ms=HOP_MS,
        )
        return _fit(values, target_frames(audio.size))


class FireRedAdapter(VadAdapter):
    """Official FireRedVAD non-streaming ONNX model.

    This one is **not causal**: it consumes the whole utterance at once, so its
    score is an upper reference and a distillation teacher, not a streaming
    competitor. It is labelled as such in every report.
    """

    hop_ms = 10.0
    causal = False
    scope = "official non-streaming model over the full clip; no future-context limit"

    def __init__(self, model_path: str | Path, cmvn_path: str | Path):
        from cadencevad.teacher import FireRedOnnxTeacher

        self.teacher = FireRedOnnxTeacher(model_path, cmvn_path)
        self.name = "fireredvad-offline"

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        values = self.teacher.predict(
            np.ascontiguousarray(audio, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
            target_hop_ms=HOP_MS,
        )
        return _fit(values, target_frames(audio.size))


class TenAdapter(VadAdapter):
    """Official TEN VAD native library driven at a 160-sample (10 ms) hop.

    TEN's licence carries additional Agora conditions. It is used here only as a
    published accuracy reference, never as a supervision source.
    """

    hop_ms = 10.0
    causal = True
    scope = "official native frontend + model + decision"

    def __init__(self, library_path: str | Path, hop_samples: int = HOP_SAMPLES):
        self.library = ctypes.CDLL(str(library_path))
        self.hop_samples = int(hop_samples)
        self.name = "ten-vad"

        self.library.ten_vad_create.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        self.library.ten_vad_create.restype = ctypes.c_int
        self.library.ten_vad_process.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.library.ten_vad_process.restype = ctypes.c_int
        self.library.ten_vad_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.library.ten_vad_destroy.restype = ctypes.c_int

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        frames = target_frames(audio.size)
        padded = np.pad(
            np.ascontiguousarray(audio, dtype=np.float32),
            (0, frames * self.hop_samples - audio.size),
        )
        pcm = np.clip(np.rint(padded * 32_768.0), -32_768, 32_767).astype(np.int16)

        handle = ctypes.c_void_p(0)
        if self.library.ten_vad_create(ctypes.byref(handle), self.hop_samples, 0.5) != 0:
            raise RuntimeError("ten_vad_create failed")
        probability = ctypes.c_float()
        flags = ctypes.c_int32()
        values = np.empty(frames, dtype=np.float32)
        try:
            for index in range(frames):
                chunk = np.ascontiguousarray(
                    pcm[index * self.hop_samples : (index + 1) * self.hop_samples]
                )
                status = self.library.ten_vad_process(
                    handle,
                    chunk.ctypes.data_as(ctypes.c_void_p),
                    self.hop_samples,
                    ctypes.byref(probability),
                    ctypes.byref(flags),
                )
                if status != 0:
                    raise RuntimeError(f"ten_vad_process failed at frame {index}")
                values[index] = probability.value
        finally:
            self.library.ten_vad_destroy(ctypes.byref(handle))
        return values


class WebrtcAdapter(VadAdapter):
    """Classic WebRTC VAD at 10 ms frames; emits hard 0/1 decisions."""

    hop_ms = 10.0
    causal = True
    scope = "official GMM decision; binary output, so AUC is a step function"

    def __init__(self, aggressiveness: int = 2):
        import webrtcvad

        self.detector = webrtcvad.Vad(aggressiveness)
        self.name = f"webrtc-vad:{aggressiveness}"

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        frames = target_frames(audio.size)
        padded = np.pad(
            np.ascontiguousarray(audio, dtype=np.float32),
            (0, frames * HOP_SAMPLES - audio.size),
        )
        pcm = np.clip(np.rint(padded * 32_768.0), -32_768, 32_767).astype(np.int16)
        values = np.empty(frames, dtype=np.float32)
        for index in range(frames):
            chunk = pcm[index * HOP_SAMPLES : (index + 1) * HOP_SAMPLES]
            values[index] = float(self.detector.is_speech(chunk.tobytes(), SAMPLE_RATE))
        return values


__all__ = [
    "FireRedAdapter",
    "CadenceVadOnnxAdapter",
    "CadenceVadTorchAdapter",
    "SileroAdapter",
    "TenAdapter",
    "VadAdapter",
    "WebrtcAdapter",
    "target_frames",
]
