#!/usr/bin/env python3
"""Assert the portable C runtime reproduces the PyTorch reference.

The embedded runtime is a hand-written reimplementation of the model, so nothing
but a numerical comparison proves the two agree. This drives the compiled library
through a fixed pseudo-random signal one 10 ms hop at a time and compares every
emitted probability against the checkpoint evaluated in PyTorch.

It is platform-agnostic on purpose: the same check gates the Linux build locally
and the MSVC build in CI, which is what makes the Windows artifact trustworthy
without a Windows host to benchmark on.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np

HOP_SAMPLES = 160
SAMPLE_RATE = 16_000


def load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.cadencevad_state_size.restype = ctypes.c_size_t
    library.cadencevad_init.argtypes = [ctypes.c_void_p]
    library.cadencevad_init.restype = ctypes.c_int
    library.cadencevad_process_hop.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.cadencevad_process_hop.restype = ctypes.c_float
    library.cadencevad_destroy.argtypes = [ctypes.c_void_p]
    return library


def native_probabilities(library: ctypes.CDLL, audio: np.ndarray) -> np.ndarray:
    state = ctypes.create_string_buffer(library.cadencevad_state_size())
    if library.cadencevad_init(state) != 0:
        raise SystemExit("cadencevad_init failed")
    hops = audio.size // HOP_SAMPLES
    values = np.empty(hops, dtype=np.float64)
    try:
        for index in range(hops):
            chunk = np.ascontiguousarray(
                audio[index * HOP_SAMPLES : (index + 1) * HOP_SAMPLES],
                dtype=np.float32,
            )
            values[index] = library.cadencevad_process_hop(
                state, chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
    finally:
        library.cadencevad_destroy(state)
    return values


def reference_probabilities(checkpoint: Path, audio: np.ndarray) -> np.ndarray:
    import torch

    from cadencevad.benchmark import load_checkpoint
    from cadencevad.features import CausalFeatureExtractor

    config, model = load_checkpoint(checkpoint)
    model = model.eval()
    frontend = CausalFeatureExtractor(config.feature).eval()
    with torch.inference_mode():
        features = frontend(torch.from_numpy(audio).unsqueeze(0))
        logits = model(features)["speech_logits"]
        return torch.sigmoid(logits).squeeze(0).double().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/cadencevad-v0.1/cadencevad-v0.1.pt"),
        help="checkpoint whose weights are embedded in the library",
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    audio = rng.normal(0.0, 0.15, int(args.seconds * SAMPLE_RATE)).astype(np.float32)

    native = native_probabilities(load_library(args.library), audio)
    reference = reference_probabilities(args.checkpoint, audio)

    usable = min(native.size, reference.size)
    if usable == 0:
        print("no frames compared", file=sys.stderr)
        return 1
    deviation = np.abs(native[:usable] - reference[:usable])

    print(f"hops compared      : {usable}")
    print(f"max abs deviation  : {deviation.max():.3e}")
    print(f"mean abs deviation : {deviation.mean():.3e}")
    print(f"tolerance          : {args.tolerance:.1e}")

    if not np.all(np.isfinite(native[:usable])):
        print("FAIL: native output contains non-finite values", file=sys.stderr)
        return 1
    if deviation.max() >= args.tolerance:
        print("FAIL: portable runtime diverges from the PyTorch reference", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
