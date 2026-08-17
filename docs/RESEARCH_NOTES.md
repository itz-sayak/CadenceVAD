# Research notes: where compact causal VAD actually stands

Background reading behind the AVA-Speech work in [`AVA_SPEECH.md`](AVA_SPEECH.md).
The purpose is narrow: establish what number CadenceVAD has to beat, under what
protocol, before claiming anything.

## The protocol matters more than the score

The single most important thing in this literature is that **published AVA-Speech
AUCs are not mutually comparable**, and the gap is large enough to invert
rankings.

Most compact VAD papers evaluate with a non-causal sliding window at 87.5%
overlap, which gives the classifier roughly 1,181 ms of total context including
future audio, and then applies median smoothing on top. A streaming VAD in a
voice agent has none of that: it sees only the past, and it must answer now.

kiloVAD (Bauer et al., INTERSPEECH 2026, arXiv:2607.25870) is the paper that
makes this explicit, and it quantifies the cost. AtomicVAD reports 0.903 AUC
under the non-causal protocol and **0.869 under causal evaluation** — the same
weights, a 0.034 swing purely from protocol. kiloVAD's Table 1 is currently the
only place several of these models are compared on a level footing.

Practical consequence: when someone quotes "0.914 AUC on AVA-Speech" for an 8k
parameter model, that is a non-causal number and it is not the bar a streaming
telephony VAD has to clear.

## The causal leaderboard

From kiloVAD Table 1, restricted to models evaluated causally:

| Model | Params | Input ctx | AUC (AVA) | Notes |
|---|---:|---:|---:|---|
| MarbleNet | 91 k | 630 ms | 0.850 | NVIDIA NeMo, 1D time-channel separable conv |
| kiloVAD (pruned) | 2.1 k | 200 ms | 0.850 | structured pruning + angle-aware QAT |
| kiloVAD (full) | 81.1 k | 200 ms | 0.862 | CNN-only, Mel frontend, TFLM-compatible |
| kiloVAD (full) | 81.1 k | 360 ms | 0.872 | same weights, longer context |
| AtomicVAD | 0.3 k | 630 ms | 0.869 | GGCU oscillatory activation |
| **ResectNet** | **4.5 k** | **200 ms** | **0.886** | **best published causal AUC** |

And the non-causal entries, for completeness — these are *not* the bar:

| Model | Params | Total ctx | AUC | Protocol |
|---|---:|---:|---:|---|
| TinyVAD | 11.6 k | 1181 ms | 0.864 | non-causal, 87.5% overlap |
| SincQDR-VAD | 8.0 k | 1181 ms | 0.914 | non-causal, 87.5% overlap |
| CNN-BiLSTM | 552 k | — | 0.948 | non-causal |
| Wav2Vec2-XLS-R | 316 M | — | 0.962 | non-causal, pretrained |

kiloVAD claims state of the art for *deployment-ready* causal VAD, a narrower
bracket than causal alone: it additionally requires a standard Mel frontend and
portable operators. ResectNet is excluded from that bracket for using a learnable
raw-audio frontend and a GRU, not because it scores worse.

**CadenceVAD sits in the causal bracket** and, being GRU-based with unbounded left
context, is not context-limited the way the 200 ms models are. So the honest
target is ResectNet's 0.886, with kiloVAD's 0.872 as the intermediate mark.

## What the strong models do that CadenceVAD did not

**Training data shape.** kiloVAD trains on LibriSpeech train-clean-100 mixed three
ways: 25% clean, 25% wind noise at −5 dB SNR, 50% DNS-Challenge noise at
{−10, −5, 0, 5, 10} dB, half of it reverberated. CadenceVAD v0.1 trained on ~3 h of
FLEURS read speech that is roughly 95% speech by duration. A model never shown
realistic silence cannot learn when to stay quiet, and its false-alarm rate says
so. This was the largest single defect and the first thing fixed here.

**Amplitude-agnostic features.** kiloVAD normalizes each Mel bin to zero mean and
unit variance across the input window, and attributes robustness across recording
conditions and microphone gains to it: the model is forced to learn relative
spectral patterns rather than absolute energy. CadenceVAD's frontend instead
subtracts each frame's mean *across* bands, which normalizes spectral tilt but
leaves per-band level drift untouched. The causal running-normalization
experiment here follows from that observation — see AVA_SPEECH.md for the result,
which was not the one expected.

**Ranking-aware objectives.** SincQDR-VAD (arXiv:2508.20885) adds a quadratic
disparity ranking loss, a squared hinge on the score gap of (speech, non-speech)
pairs, combined as `0.25 * L_QDR + 0.75 * L_BCE` with margin 1.0. The motivation
applies directly here: AUC is a pure ranking metric and cross entropy optimizes
calibration instead. Implemented as `cadencevad.losses.pairwise_ranking_loss`.

**PCEN.** Per-channel energy normalization replaces static log compression with
per-band automatic gain control and stabilized root compression, and is causal by
construction. It is well supported for far-field and noisy speech. It is *not*
implemented here, for a concrete engineering reason: PCEN's smoother conventionally
initializes to the first frame's energy, and this repository's streaming contract
zero-initializes every state tensor in both the ONNX runtime and the embedded C
runtime. Making PCEN correct under zero-init needs either a warm-up counter in the
state or a modified formulation. Recorded as future work rather than bodged.

## Teachers and supervision

FireRedVAD (Apache-2.0, FireRedTeam) reports 97.57% F1 and 99.60% AUC on
FLEURS-VAD-102, against Silero at 95.95/97.99 and TEN-VAD at 95.19/97.81. It is a
DFSMN-based model with a non-streaming mode that consumes the whole utterance.
That makes it a strong non-causal teacher for a causal student, which is the
classic setup for getting a small streaming model above its weight class. This
repository already contained a `FireRedOnnxTeacher` adapter that nothing used; the
mixture builder now uses it, ensembled with Silero.

FLEURS-VAD-102 itself would be a clean second test set — CadenceVAD's FLEURS usage
is confined to `train` and `validation`, so FLEURS `test` is genuinely untouched —
but the annotations are still listed as "coming soon" and were not available.

TEN-VAD is used here only as a runtime and accuracy reference. Its licence carries
additional Agora conditions, and this repository's own `docs/DATA.md` already
forbids using its outputs as a supervision source.

## Acoustic VAD is not turn detection

The LinkedIn exchange that prompted this work makes a point worth recording:
a per-window speech probability is not an end-of-turn signal. Both parties in that
thread converge on it, and the practitioner reply is the sharpest summary — the
one reliable standalone-VAD use case is fast barge-in detection, because it beats
a round trip through ASR; everything else needs a hybrid.

The current generation of turn detectors reflects this. Pipecat's Smart Turn v2
predicts turn completion from the raw waveform (wav2vec2 + linear head,
14 languages), and LiveKit's turn detector moved from a text-only transformer over
ASR output to an audio model encoding user audio directly. Both sit *above* a VAD
rather than replacing it.

So the work here deliberately optimizes acoustic VAD quality and latency, and does
not claim to address endpointing. The integration point is the hysteresis
detector's `speech_start`/`speech_end` events, which a higher-level policy should
combine with ASR stability, semantic completion, interruption state and timeouts.

## References

- Bauer et al., *VAD to the Bone: Ultra-Tiny Speech Activity Detection for Edge
  Deployment*, INTERSPEECH 2026. arXiv:2607.25870
- Wang et al., *SincQDR-VAD: A Noise-Robust Voice Activity Detection Framework
  Leveraging Learnable Filters and Ranking-Aware Optimization*. arXiv:2508.20885
- Jia, Majumdar, Ginsburg, *MarbleNet: Deep 1D Time-Channel Separable
  Convolutional Neural Network for Voice Activity Detection*. arXiv:2010.13886
- Köpüklü & Taseska, *ResectNet: An Efficient Architecture for Voice Activity
  Detection on Mobile Devices*, INTERSPEECH 2022, pp. 5363–5367
- Chae & Lee, *Small-footprint convolutional neural network with reduced feature
  map for voice activity detection*, ICASSP 2024, pp. 12266–12270
- Soto-Vergel et al., *AtomicVAD: A tiny voice activity detection model for
  efficient inference in intelligent IoT systems*, Internet of Things, 2025
- Chaudhuri et al., *AVA-Speech: A Densely Labeled Dataset of Speech Activity in
  Movies*, INTERSPEECH 2018. arXiv:1808.00606
- Wang et al., *Trainable Frontend For Robust and Far-Field Keyword Spotting*
  (PCEN); Lostanlen et al., *Per-Channel Energy Normalization: Why and How*,
  IEEE SPL 2019
- FireRedTeam, *FireRedVAD* / *FireRedASR2S*. arXiv:2603.10420
- Silero Team, *Silero VAD*; TEN Framework, *TEN VAD*
