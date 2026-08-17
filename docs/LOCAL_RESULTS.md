# CadenceVAD validation record

Date: 2026-08-17

## Release classification

**Software status:** the runtime, packaging, adapters and evaluation harness are
reproducible and tested (162 tests).

**Checkpoint status:** research preview, not production-ready. The best model here
reaches 0.8436 AUC on AVA-Speech against a published causal state of the art of
0.886, and against Silero at 0.9022 measured on the same harness.

## Measured machines

**Training and evaluation:** AMD Ryzen Threadripper PRO 7965WX (24 cores),
251 GB RAM, NVIDIA RTX 4090 (24 GB, driver 570.86.10), Linux 6.5,
Python 3.13.12, torch 2.13.0+cu126, onnxruntime 1.24.4.

**Runtime latency:** the same host, single thread, pinned core, machine otherwise
idle. Recorded in `artifacts/environment.json`.

## Runtime latency

Complete causal frontend plus model, one 10 ms hop, 50,000 iterations after 5,000
warm-up.

| Build | p50 | p95 | p99 | mean | RTF (p50) |
|---|---:|---:|---:|---:|---:|
| portable + AVX2 | **9.024 µs** | 9.054 µs | 9.834 µs | 9.047 µs | 0.00090 |
| portable, scalar | 27.061 µs | 28.333 µs | 30.206 µs | 27.214 µs | 0.00271 |

Per-call state is 20,064 bytes. An Apple Accelerate baseline runtime measured
11.417 µs p50 on an M4 Pro; the portable AVX2 build is faster while depending on
nothing beyond a C11 compiler.

Artifacts: `benchmarks/portable-runtime/linux-x86_64-{avx2,scalar}.json`.

### Numerical parity

| Path | Reference | Max deviation |
|---|---|---:|
| portable C (AVX2) | PyTorch | 9.686e-08 |
| portable C (scalar) | PyTorch | 1.341e-07 |
| portable FFT power spectrum | NumPy `rfft` | 1.35e-06 relative |
| ONNX streaming | PyTorch offline | 2.98e-07 |
| offline-batched evaluator | streaming ONNX | 1.9e-04 |

The last figure is the same magnitude as CPU-versus-GPU float noise, confirming
the batched evaluator used for long clips is the streaming computation.

### Windows

The identical source builds with MSVC via `native/portable/CMakeLists.txt`, and CI
runs a `windows-latest` matrix (AVX2 and scalar) asserting PyTorch parity to 1e-4.
**No Windows timings are reported.** No Windows host was available; estimating them
from Linux would be misleading. Windows correctness is proven, Windows speed is
open.

## AVA-Speech

Full benchmark, 160 clips, 39.667 h (99.2% of the 40 h nominal), strictly causal,
10 ms frames, no smoothing, no future context, 1,000-iteration clip-level
bootstrap. Protocol: `docs/AVA_SPEECH.md`.

| Model | Params | AUC | 95% CI | F1 | FA | Miss | clean | music | noise |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Silero VAD | — | 0.9022 | [0.8910, 0.9127] | 0.839 | 0.140 | 0.186 | 0.956 | 0.859 | 0.895 |
| TEN VAD | — | 0.8966 | [0.8905, 0.9026] | 0.828 | 0.177 | 0.180 | 0.910 | 0.896 | 0.889 |
| conv-v3-m-rank | 99,930 | 0.8436 | [0.8242, 0.8494] | 0.777 | 0.164 | 0.270 | 0.909 | 0.806 | 0.827 |
| conv-v1-base | 46,170 | 0.8418 | [0.8300, 0.8536] | 0.765 | 0.157 | 0.292 | 0.912 | 0.796 | 0.827 |
| conv-v3-m | 99,930 | 0.8348 | [0.8180, 0.8427] | 0.773 | 0.189 | 0.262 | 0.906 | 0.785 | 0.821 |
| conv-v1-ema | 49,008 | 0.8300 | [0.8148, 0.8441] | 0.725 | 0.091 | 0.385 | 0.882 | 0.779 | 0.828 |
| v0.1 baseline | 46,170 | 0.8295 | [0.8187, 0.8408] | 0.764 | 0.358 | 0.181 | 0.861 | 0.814 | 0.820 |
| conv-v1-ema-keepmean | 49,008 | 0.8085 | [0.7927, 0.8194] | 0.681 | 0.088 | 0.442 | 0.873 | 0.760 | 0.799 |
| conv-v2-rank-m | 99,930 | 0.8058 | [0.7793, 0.8054] | 0.723 | 0.155 | 0.354 | 0.871 | 0.769 | 0.789 |
| conv-v3-base | 46,170 | 0.8024 | [0.7883, 0.8113] | 0.733 | 0.252 | 0.289 | 0.864 | 0.777 | 0.781 |
| conv-v2-rank | 46,170 | 0.7864 | [0.7598, 0.7874] | 0.688 | 0.143 | 0.408 | 0.846 | 0.746 | 0.774 |
| conv-v2-base | 46,170 | 0.7833 | [0.7667, 0.7899] | 0.707 | 0.205 | 0.352 | 0.835 | 0.749 | 0.773 |

The Silero and TEN rows are our own measurements. Strictly causal AVA-Speech
numbers for either model do not appear to be published; they are supplied as
reference points.

Published causal reference points (kiloVAD arXiv:2607.25870 Table 1, not measured
here): ResectNet 0.886, kiloVAD 0.872 at 360 ms context, AtomicVAD 0.869,
kiloVAD 0.862 at 200 ms, MarbleNet 0.850.

## Established effects

### Training duty cycle

Retraining the unchanged baseline architecture on 44 h of call-shaped mixtures with a
realistic speech duty cycle (~39%, versus FLEURS's ~95%):

| | v0.1 baseline | conv-v1-base | Δ |
|---|---:|---:|---:|
| false-alarm rate | 0.358 | 0.157 | −0.201 |
| clean-speech AUC | 0.861 | 0.912 | +0.051 |
| overall AUC | 0.8295 | 0.8418 | +0.012 |

At roughly 16 standard errors the false-alarm reduction is unambiguous. Six
corpora spanning target duty 0.15–0.90 are built for the full sweep; the sweep
itself has not been run.

### Aggressive mixing regression

`mixtures-v2` (SNR floor −12 dB, doubled music weight, noise probability 0.92,
RIR 0.45, 108 h, 1,212 speakers) scored 0.7833 against 0.8418 — a 0.058 loss, 4.6
standard errors. **Confounded**: four variables changed simultaneously. The
leading hypothesis is that labelling from clean source audio while mixing below
intelligibility injects label noise, but that is untested. The SNR-floor sweep to
isolate it has not been run.

## Reproducibility floor

Five runs differing only in random seed:

| Setting | n | mean | sd | range |
|---|---:|---:|---:|---:|
| 40 epochs, best-dev selection, 44 h | 5 | 0.8069 | 0.0127 | 0.0302 |
| 200 epochs, Polyak averaging, 20 h | 5 | 0.7101 | 0.0322 | 0.0812 |

Development F1 varies by sd 0.0008 across the first group while AVA AUC varies by
0.0127. The dev split is saturated, so best-dev epoch selection is close to
random.

The second row was an attempt to fix this and made both mean and variance worse.
It is **confounded** — training length, averaging and the dataset all changed
together — and has not been isolated.

This variance is roughly an order of magnitude above the ±0.001 kiloVAD reports
over 10 seeds, which indicts this training recipe rather than the field.

**Consequence:** effects below about 0.02 AUC are not resolvable at n=3 here.
Several single-seed comparisons in this repository's history — the ranking loss
(+0.003), the running-normalisation input stage (−0.012), the capacity increase
(+0.022) — are **not conclusions**.

## Modulation filterbank

Three seeds per arm, matched data and config:

| Metric | base | + filterbank | Δ | SE |
|---|---:|---:|---:|---:|
| overall AUC | 0.8056 | 0.8103 | +0.005 | 0.017 |
| music AUC | 0.7634 | 0.7797 | +0.016 | 0.013 |
| clean AUC | 0.8786 | 0.8596 | −0.019 | 0.020 |

Not significant at this sample size. See `docs/MODULATION.md`.

## Not done

- duty-cycle sweep (corpora built, sweep not run);
- SNR-floor sweep;
- kiloVAD recipe replication as a controlled baseline;
- modulation filterbank in the C runtime;
- multi-timescale recurrence;
- a stabilised training recipe, which gates everything above.

## Release decision

The repository is publishable as a runtime, harness and reproducible study. The
checkpoints must continue to be labelled a research preview. No accuracy claim
beyond the tables above is supported.
