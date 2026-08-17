---
license: cc-by-4.0
library_name: cadencevad
language:
  - ar
  - en
  - gu
  - hi
  - kn
  - pa
  - ta
  - te
  - ur
tags:
  - audio
  - voice-activity-detection
  - vad
  - streaming
  - onnx
  - pytorch
  - telephony
  - webrtc
---

# CadenceVAD model card

## Summary

CadenceVAD is a 46,170-parameter causal streaming voice-activity detector
(47,502 with the modulation filterbank). It consumes 16 kHz mono audio, produces
one speech probability every 10 ms, and keeps independent convolutional,
recurrent, feature and detector state per call.

**Author:** Sayak Dutta

**Status: research preview.** These checkpoints are for integration testing,
browser demonstration and shadow evaluation. They are **not approved for
production**. The best model measured here reaches 0.8436 ROC-AUC on AVA-Speech
against a published causal state of the art of 0.886, and against Silero at
0.9022 measured on the same harness.

### Provenance of the bundled `cadencevad-v0.1` artifacts

The `cadencevad-v0.1` files shipped in this repository are **not
CadenceVAD-trained**. They are the FlashVAD v0.1 checkpoint by Himanshu Maurya,
redistributed unmodified under CC BY 4.0 and retained only as a comparison
baseline. See [`NOTICE`](NOTICE) and
[`MODEL_LICENSE.md`](MODEL_LICENSE.md). CadenceVAD-trained checkpoints are
produced by `scripts/run_experiment_sweep.py` and are not distributed as release
artifacts yet.

## Measured accuracy

Full AVA-Speech, 39.667 h, strictly causal, 10 ms frames, no smoothing, with
1,000-iteration clip-level bootstrap intervals. Protocol in
[`docs/AVA_SPEECH.md`](docs/AVA_SPEECH.md).

| Model | Params | ROC-AUC | 95% CI | FA | Miss | clean | music | noise |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| CadenceVAD conv-v3-m-rank | 99,930 | 0.8436 | [0.8242, 0.8494] | 0.164 | 0.270 | 0.909 | 0.806 | 0.827 |
| CadenceVAD conv-v1-base | 46,170 | 0.8418 | [0.8300, 0.8536] | 0.157 | 0.292 | 0.912 | 0.796 | 0.827 |
| v0.1 reference (upstream) | 46,170 | 0.8295 | [0.8187, 0.8408] | 0.358 | 0.181 | 0.861 | 0.814 | 0.820 |
| *Silero VAD, same harness* | — | *0.9022* | *[0.8910, 0.9127]* | *0.140* | *0.186* | *0.956* | *0.859* | *0.895* |
| *TEN VAD, same harness* | — | *0.8966* | *[0.8905, 0.9026]* | *0.177* | *0.180* | *0.910* | *0.896* | *0.889* |

Run-to-run seed variance on this benchmark is sd 0.0127, so differences below
about 0.02 AUC are not resolvable. Speech-with-music is the weakest condition
throughout.

## Runtime latency

Idle AMD Ryzen Threadripper PRO 7965WX, Linux x86-64, single thread, pinned core.
Complete causal frontend plus model for one 10 ms hop.

| Build | p50 | p95 | p99 | RTF |
|---|---:|---:|---:|---:|
| portable + AVX2 | 9.02 µs | 9.05 µs | 9.83 µs | 0.00090 |
| portable, scalar | 27.06 µs | 28.33 µs | 30.21 µs | 0.00271 |

Windows builds from the same source and is parity-checked in CI; Windows timings
are not measured.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| `cadencevad-v0.1.pt` | `ca9e35475518466b2a1f2e89b4953cd1e26e3d8c513cdcf265ab319e74e2b288` |
| `cadencevad-stream.onnx` | `9a88e34bf3118d60e25a16cb622cb394e2f3ab71445b0aa5957df6f1d5f1b6ba` |
| `config.json` | `0b1ad372808f7c67cea5a1ca4b41a817714c9a0b0cf49b3baa56fe8d5f64ad2b` |
| `detector-calibration.json` | `b5d000e0406d81fbd87a9e66194a877fa8433a76783faac8275c7969c43051b4` |

The ONNX graph accepts precomputed 43-dimensional causal features. Use
`src/cadencevad/features.py` or
`report-site/src/lib/vad-features.mjs`; it is not a raw-waveform graph.

The public ONNX file is self-contained and stripped of exporter stack traces,
local paths, and private build metadata.

## Download and source

Download the complete model repository:

```bash
hf download itz-sayak/CadenceVAD --local-dir cadencevad-model
```

Source code and runtime integrations are published separately at
[`itz-sayak/CadenceVAD`](https://github.com/itz-sayak/CadenceVAD).

The ONNX graph does not accept raw waveform audio. It expects the causal
43-dimensional features described below, with independent feature and model
state for every call.

## Intended use

Appropriate current uses:

- research and architecture evaluation;
- functional integration with browser, LiveKit, Pipecat, SIP, or PSTN stacks;
- latency and concurrency measurement;
- shadow-mode comparison on consented, labelled call audio.

Do not use this checkpoint as the sole basis for:

- emergency, medical, legal, financial, or safety-critical decisions;
- call recording consent or compliance decisions;
- a production multilingual accuracy claim;
- semantic end-of-turn detection.

Reset all feature, model, resampler, and detector state when a call ends or an
audio discontinuity occurs.

## Architecture

- 25 ms causal analysis frame and 10 ms hop;
- 40 log-mel bands plus energy, zero-crossing rate, and spectral flatness;
- four causal depthwise temporal blocks with dilations 1, 2, 4, and 8;
- one 64-unit GRU;
- speech and auxiliary event heads;
- 184,680 bytes of FP32 parameters.

## Training inputs and provenance limit

Historical training notes associated with the retained checkpoint report:

- 558 derived clips from nine FLEURS configurations: Arabic, English,
  Gujarati, Hindi, Kannada, Punjabi, Tamil, Telugu, and Urdu;
- 288 AMI meeting clips with meeting-family-disjoint train/validation splits;
- 64 MUSAN noise clips;
- weak frame targets from the official Silero VAD model.

FLEURS, AMI, and MUSAN attribution is in `NOTICE`; dataset audio is not
distributed here.

Those counts and corpus names are not embedded as complete provenance in the
public checkpoint. The exact retained training manifests, their digests, source
revisions, and teacher-output digest were not preserved, so the checkpoint's
training run is **not bit-for-bit reproducible** from the public tree. This is a
provenance limitation, not evidence of broader accuracy. Future release
candidates must preserve those records before training begins.

## Evaluation

Primary evaluation is the full AVA-Speech benchmark under a strictly causal
protocol; see [Measured accuracy](#measured-accuracy) above and
[`docs/AVA_SPEECH.md`](docs/AVA_SPEECH.md). AVA-Speech is held out: it is never
used for training, distillation targets, hard-negative mining or threshold
selection, and `scripts/run_experiment_sweep.py` selects on the synthetic
development split only.

### Secondary: TEN VAD public set

The upstream v0.1 reference checkpoint was repeatedly inspected on TEN VAD's
public 30-recording set during its original development, giving 0.882 ROC-AUC,
0.889 hysteresis-decision F1, a 26.3% false-alarm rate and a 13.0% miss rate over
26,243 frames. Because that set influenced research decisions upstream, those are
exploratory external-set numbers, not an untouched test. Language is recorded as
`und`; codec, channel, device and SNR are unknown.

Machine-readable report:
`benchmarks/cadencevad-v0.1/ten-public-evaluation.json`.

### Interpreting differences

Run-to-run seed variance on AVA-Speech is sd 0.0127 with this training recipe,
while development-split F1 varies by only sd 0.0008. **Differences below roughly
0.02 AUC are not resolvable at three seeds** and should be treated as unresolved
rather than as results. See
[`docs/LOCAL_RESULTS.md`](docs/LOCAL_RESULTS.md#reproducibility-floor).

## Known limitations

- Quiet speech, music, TTS leakage, echo, television, laughter, singing, and
  overlapping speakers may cause misses or false triggers.
- Read speech and meetings do not cover real carrier, device, packet-loss, and
  room conditions.
- Language presence in training does not prove per-language performance.
- Linear 8-to-16 kHz conversion prioritizes causal speed, not audio fidelity.
- VAD cannot determine whether a speaker has semantically completed a turn.

A production candidate needs consented, human-labelled, speaker-disjoint calls
with predeclared per-slice gates and a test set untouched until final
evaluation.

## Licences

Repository source code is MIT-licensed. The retained model artifacts are
separately available under CC BY 4.0; see
`MODEL_LICENSE.md` and `NOTICE`. Third-party datasets, models, and benchmark
materials retain their own terms.
