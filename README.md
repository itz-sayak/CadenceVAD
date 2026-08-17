# CadenceVAD

CadenceVAD is a small causal voice-activity detector for low-latency voice calls.
It emits one speech probability every 10 ms and ships with a portable C runtime
that runs the whole frontend and model in **9.02 µs per hop** on one x86-64 core.

Created by **Sayak Dutta**.

The name refers to speech cadence: the model carries a learnable
modulation-domain filterbank tuned to the 4–8 Hz syllabic rate that separates
speech from sustained music and stationary noise.

> **Status: research preview.** The runtime, packaging and evaluation harness are
> production-quality and reproducible. The **checkpoints are not** — the best
> model here reaches 0.844 AUC on AVA-Speech against a published causal state of
> the art of 0.886. Read [Results](#results) before making any accuracy claim.

## What is here

- a 46,170-parameter streaming model (47,502 with the modulation filterbank);
- a **portable C runtime** for Windows, Linux and macOS — no Apple Accelerate
  dependency, AVX2 with a scalar fallback, verified against PyTorch to 9.7e-8;
- a **single-protocol AVA-Speech harness** over the full 39.67 h benchmark, with
  causal numbers for Silero and TEN that are not published elsewhere;
- a conversation-mixture data pipeline with frame-exact labels;
- a shared-session ONNX runtime with independent per-call state;
- LiveKit Agents and Pipecat adapters; PCMU, PCMA and PCM16 telephone ingress.

## Install

```bash
git clone https://github.com/itz-sayak/CadenceVAD.git
cd CadenceVAD
pip install .
```

Development:

```bash
uv sync --all-extras
uv run pytest          # 162 tests
uv run ruff check .
```

## Runtime latency

Measured on an **idle AMD Ryzen Threadripper PRO 7965WX**, Linux x86-64, single
thread, pinned core, 50,000 iterations after 5,000 warm-up. Scope is the complete
causal frontend plus the model for one 10 ms hop.

| Build | p50 | p95 | p99 | Real-time factor | State |
|---|---:|---:|---:|---:|---:|
| **portable + AVX2** | **9.02 µs** | 9.05 µs | 9.83 µs | 0.00090 | 20,064 B |
| portable, scalar fallback | 27.06 µs | 28.33 µs | 30.21 µs | 0.00271 | 20,064 B |

For reference, an Apple Accelerate baseline runtime measured 11.417 µs p50 on an
M4 Pro. The portable AVX2 build is faster than that while depending on nothing but
a C11 compiler.

At a 0.00090 real-time factor a single core sustains roughly 1,100 concurrent
streams before saturation, though a real deployment should size from p99.

Artifacts: [`benchmarks/portable-runtime/`](benchmarks/portable-runtime/).

**Windows.** The same source builds with MSVC through
[`native/portable/CMakeLists.txt`](native/portable/CMakeLists.txt), and CI runs a
`windows-latest` job that asserts numerical parity with PyTorch to 1e-4 for both
the AVX2 and scalar builds. **Windows timings are not reported here** — no Windows
host was available to measure them, and estimating them from Linux would be
dishonest. Correctness on Windows is proven; speed on Windows is an open item.

**End-to-end latency** is dominated by the algorithm, not the arithmetic: a 25 ms
analysis window plus `start_frames × 10 ms` of detector persistence puts
`speech_start` about 55 ms after speech onset with the default calibration, of
which `pre_roll_frames` recovers 30 ms.

## Results

Every number below comes from
[`scripts/benchmark_ava_speech.py`](scripts/benchmark_ava_speech.py) on the full
39.67 h of AVA-Speech (99.2% of the 40 h benchmark), strictly causal, 10 ms
frames, no smoothing, no future context. Intervals are 1,000-iteration
clip-level bootstraps. Protocol details: [`docs/AVA_SPEECH.md`](docs/AVA_SPEECH.md).

| Model | Params | AUC | 95% CI | F1 | FA | Miss | clean | music | noise |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Silero VAD | — | 0.9022 | [0.891, 0.913] | 0.839 | 0.140 | 0.186 | 0.956 | 0.859 | 0.895 |
| TEN VAD | — | 0.8966 | [0.891, 0.903] | 0.828 | 0.177 | 0.180 | 0.910 | 0.896 | 0.889 |
| CadenceVAD conv-v3-m-rank | 99,930 | 0.8436 | [0.824, 0.849] | 0.777 | 0.164 | 0.270 | 0.909 | 0.806 | 0.827 |
| **CadenceVAD conv-v1-base** | 46,170 | **0.8418** | [0.830, 0.854] | 0.765 | **0.157** | 0.292 | 0.912 | 0.796 | 0.827 |
| v0.1 baseline checkpoint | 46,170 | 0.8295 | [0.819, 0.841] | 0.764 | 0.358 | 0.181 | 0.861 | 0.814 | 0.820 |

The Silero and TEN rows are our own measurements. I could not find strictly causal
AVA-Speech numbers for either model published anywhere, and they are supplied here
as a reference point for the compact-VAD literature.

### Published causal reference points

Not measured here; from kiloVAD (arXiv:2607.25870) Table 1. Non-causal numbers are
excluded because they are not comparable — AtomicVAD alone swings 0.903 → 0.869
between the two protocols.

| Model | Params | AUC |
|---|---:|---:|
| ResectNet | 4.5 k | 0.886 |
| kiloVAD (360 ms context) | 81.1 k | 0.872 |
| AtomicVAD | 0.3 k | 0.869 |
| kiloVAD (200 ms context) | 81.1 k | 0.862 |
| MarbleNet | 91 k | 0.850 |

**CadenceVAD does not reach these.** Best here is 0.8436 against 0.886.

### What actually moved the needle

Training on call-shaped audio instead of read speech is the one large, unambiguous
effect measured so far. Retraining the unchanged architecture on 44 h of mixtures
with a realistic speech duty cycle (~39% rather than FLEURS's ~95%):

| | v0.1 baseline | conv-v1-base | Δ |
|---|---:|---:|---:|
| false-alarm rate | 0.358 | **0.157** | **−0.201** |
| clean-speech AUC | 0.861 | **0.912** | +0.051 |
| overall AUC | 0.8295 | 0.8418 | +0.012 |

At roughly 16 standard errors, the false-alarm reduction is far outside run-to-run
noise.

### Reproducibility floor

Five identical runs differing only in random seed:

| Setting | n | mean AUC | sd | range |
|---|---:|---:|---:|---:|
| 40 epochs, best-dev selection | 5 | 0.8069 | **0.0127** | 0.0302 |
| 200 epochs, Polyak averaging | 5 | 0.7101 | 0.0322 | 0.0812 |

Development-split F1 varies by sd 0.0008 across the same runs while AVA-Speech AUC
varies by 0.0127 — the dev split is saturated, so selecting the best-dev epoch is
close to drawing a checkpoint at random.

Two consequences worth stating plainly. **Any effect smaller than about 0.02 AUC
is not resolvable here at n=3**, so several single-seed comparisons in this
repository's history are not conclusions. And this variance is roughly an order of
magnitude larger than kiloVAD's reported ±0.001 over 10 seeds, which points at
this training recipe rather than at the field.

### Modulation filterbank

The learnable causal modulation filterbank ([`docs/MODULATION.md`](docs/MODULATION.md))
adds 12 parameters, is provably stable and zero-init-safe, and is verified
frequency-selective. Ablated at 3 seeds per arm on matched data and config:

| Metric | base | + filterbank | Δ | significance |
|---|---:|---:|---:|---|
| overall AUC | 0.8056 | 0.8103 | +0.005 | 0.3 SE — not significant |
| **music AUC** | 0.7634 | 0.7797 | +0.016 | 1.3 SE — not significant |
| clean AUC | 0.8786 | 0.8596 | −0.019 | 0.9 SE — not significant |

The music effect points the way the design predicts and is the largest of the
three, but it does not clear significance at this sample size. **It is not
currently a demonstrated improvement**, and resolving it would need roughly 12
seeds per arm.

## Usage

Create one process-level ONNX owner and one small stream per call:

```python
from cadencevad.runtime import OnnxStreamingVadModel

model = OnnxStreamingVadModel.load_bundled(threads=1)
call = model.new_stream()
probabilities, events = call.push(audio_float32_16khz)
call.reset()
```

Never load an ONNX session inside the per-packet or per-call hot path. The bundled
loader verifies digests, validates the tensor contract, and rejects silent
execution-provider fallback, so a CUDA request cannot quietly run on CPU.

Telephone audio, decoded and causally resampled from 8 kHz G.711:

```python
from cadencevad.telephony import TelephonyVadStream

call = TelephonyVadStream.load_onnx("pcmu")
probabilities, events = call.push(payload_bytes)
```

LiveKit and Pipecat adapters are in
[`src/cadencevad/integrations/`](src/cadencevad/integrations/).

## Scope

CadenceVAD is **acoustic VAD, not semantic end-of-turn detection**. A per-window
speech probability cannot decide whether a speaker has finished. Its reliable
standalone use is fast barge-in detection, where it beats a round trip through
ASR. For turn-taking, combine its `speech_start`/`speech_end` events with ASR
stability, semantic completion, interruption state and a timeout policy.

## Documentation

- [`docs/AVA_SPEECH.md`](docs/AVA_SPEECH.md) — benchmark protocol and reproduction
- [`docs/MODULATION.md`](docs/MODULATION.md) — the modulation filterbank
- [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) — literature and where this sits
- [`docs/LOCAL_RESULTS.md`](docs/LOCAL_RESULTS.md) — full validation record
- [`docs/DATA.md`](docs/DATA.md) — data requirements and labelling policy
- [`MODEL_CARD.md`](MODEL_CARD.md) — intended use and limitations

## Licences

Source code is MIT. Model artifacts are CC BY 4.0; see
[`MODEL_LICENSE.md`](MODEL_LICENSE.md) and [`NOTICE`](NOTICE). The bundled
`cadencevad-v0.1` checkpoint is a third-party baseline redistributed under
CC BY 4.0, not a CadenceVAD-trained model. Third-party datasets, models and
benchmarks retain their own terms.
