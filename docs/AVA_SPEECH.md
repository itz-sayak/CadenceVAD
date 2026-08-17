# AVA-Speech evaluation

AVA-Speech is the benchmark this work measures against, because it is the only
public VAD set with a published **strictly causal** leaderboard. Everything below
describes how to reproduce the numbers and what they do and do not mean.

The comparison targets, and the reason the causal/non-causal distinction is not a
footnote, are in [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md).

## The set

AVA-Speech densely annotates speech activity in 160 fifteen-minute clips drawn
from AVA v1.0 movies. Every instant carries one of four labels: `NO_SPEECH`,
`CLEAN_SPEECH`, `SPEECH_WITH_MUSIC`, `SPEECH_WITH_NOISE`.

- **Labels**: Google's `ava_speech_labels_v1.csv`, CC BY 4.0. 39,874 segments over
  160 video ids, covering seconds 900–1800 of each source video.
- **Audio**: the Apache-2.0 Hugging Face mirror `nccratliri/vad-human-ava-speech`,
  16 kHz mono.

Neither is redistributed by this repository.

### Verifying the audio against the labels

The mirror's clips start at the beginning of the annotated AVA segment, so mirror
second `t` is official second `t + 900`. This is not documented anywhere — it was
established empirically and is then **enforced**: `prepare_ava_speech.py` computes
per-clip frame agreement between the mirror's own annotations and the official
labels at that offset and refuses to write a manifest if any clip falls below a
threshold.

On the retained run, agreement was **0.9995 minimum and 0.9998 mean across all
160 clips**. For contrast, the same comparison at offsets of 1000 s through
1500 s lands between 0.38 and 0.51, so the alignment is unambiguous.

### Realised coverage

158 of the 160 mirror clips carry the full 900 s segment; two carry a 300 s
excerpt. Realised audio is **39.67 h against a 40 h nominal benchmark (99.2%)**,
and `PROVENANCE.json` records `audio_hours`, `full_length_clips` and
`truncated_clips` so any report can state the realised figure rather than assume
the nominal one.

Frame counts at a 10 ms hop:

| Class | Frames | Share |
|---|---:|---:|
| `NO_SPEECH` | 6,796,224 | 47.6% |
| `CLEAN_SPEECH` | 2,010,616 | 14.1% |
| `SPEECH_WITH_MUSIC` | 1,907,386 | 13.4% |
| `SPEECH_WITH_NOISE` | 3,565,595 | 25.0% |
| uncovered | 179 | <0.01% |

The 179 uncovered frames fall outside every annotated segment and are excluded
from every metric rather than guessed at.

## Protocol

- Frame-level speech vs non-speech at a **10 ms** hop.
- Positives are the three speech conditions pooled; negatives are `NO_SPEECH`.
- **Each clip is streamed end to end with a single model state.** Frames are
  sliced by condition only *after* inference. Cutting the audio per condition
  first would hand a causal model a fresh zero state at every boundary and
  inflate its score.
- Per-condition AUC pits one speech condition against the same full `NO_SPEECH`
  pool.
- No smoothing, no future context, no sliding-window overlap.
- The operating threshold is the best-F1 point on the pooled subset, then reused
  unchanged for every per-condition subset.
- Confidence intervals come from a 1,000-iteration clip-level bootstrap. AUC per
  replicate is computed from per-clip score histograms, which is exact to within
  2e-7 of a direct computation and avoids re-sorting 14 M frames a thousand times.

Models with a coarser native hop are interpolated onto the 10 ms grid by the
existing `cadencevad.teacher.interpolate_probabilities` (Silero runs at 32 ms).

## Reproducing

```bash
uv run python scripts/prepare_ava_speech.py \
  --output /path/to/ava-speech --relative-paths --workers 12

PYTHONPATH=scripts uv run python scripts/benchmark_ava_speech.py \
  --dataset /path/to/ava-speech \
  --model cadencevad-torch --checkpoint models/cadencevad-v0.1/cadencevad-v0.1.pt \
  --device cuda --output benchmarks/ava-speech/baseline-cadencevad-v0.1.json
```

External baselines need their official checkouts; nothing is vendored:

```bash
# Silero
--model silero --silero-model .../silero-vad/src/silero_vad/data/silero_vad.onnx
# TEN (Agora licence terms apply)
--model ten --ten-library .../ten-vad/lib/Linux/x64/libten_vad.so
# FireRedVAD, non-streaming: a reference and teacher, NOT a causal competitor
--model firered --firered-model .../fireredvad_vad.onnx --firered-cmvn .../cmvn.ark
```

`scripts/vad_adapters.py` puts every model on the same 10 ms grid and records
whether it was evaluated causally in the output JSON.

### The CadenceVAD path is offline-batched, not a shortcut

`CadenceVadTorchAdapter` runs the convolution stack over the whole clip and feeds
the GRU in chunks while carrying its hidden state. Because every convolution pads
on the left only and the GRU starts from zeros, this is the same computation as
the 10 ms streaming loop — just batched, so a 900 s clip is one GPU call instead
of 90,000 Python steps. Verified against the shipped streaming ONNX runtime at
max |Δ| = 1.9e-4, the same magnitude as CPU-versus-GPU float noise.

## Discipline

**AVA-Speech is evaluation-only.** It is never used for training, distillation
targets, hard-negative mining, or threshold selection. `PROVENANCE.json` states
this and `scripts/run_training_sweep.py` rejects its identifiers alongside the
TEN, Silero and FireRed benchmark sets.

Training and validation data are speaker-disjoint by construction, and the
mixture generator records the exact speaker split.

One methodological caveat is worth stating plainly, because it shaped the
conclusions here: **the in-domain synthetic validation split is a poor predictor
of AVA-Speech performance.** One change in this work improved dev F1 while
lowering AVA AUC. Any candidate selected on dev alone should be treated as
unproven until measured on a genuinely different domain — which is the same trap
the v0.1 multilingual candidate fell into, where lower false alarms came with a
detector-F1 and miss-rate regression.

## Results

Full benchmark, strictly causal, 10 ms frames, no smoothing.

| Model | Params | AUC | 95% CI | FA | Miss | clean | music | noise |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Silero VAD | — | 0.9022 | [0.8910, 0.9127] | 0.140 | 0.186 | 0.956 | 0.859 | 0.895 |
| TEN VAD | — | 0.8966 | [0.8905, 0.9026] | 0.177 | 0.180 | 0.910 | 0.896 | 0.889 |
| CadenceVAD conv-v3-m-rank | 99,930 | 0.8436 | [0.8242, 0.8494] | 0.164 | 0.270 | 0.909 | 0.806 | 0.827 |
| CadenceVAD conv-v1-base | 46,170 | 0.8418 | [0.8300, 0.8536] | 0.157 | 0.292 | 0.912 | 0.796 | 0.827 |
| v0.1 baseline checkpoint | 46,170 | 0.8295 | [0.8187, 0.8408] | 0.358 | 0.181 | 0.861 | 0.814 | 0.820 |

The Silero and TEN rows are measurements made here. Strictly causal AVA-Speech
numbers for either do not appear in the literature I reviewed, and both land above
every compact model in kiloVAD's causal table.

Published causal reference points, not measured here (kiloVAD arXiv:2607.25870,
Table 1): ResectNet 0.886, kiloVAD 0.872 at 360 ms, AtomicVAD 0.869, kiloVAD 0.862
at 200 ms, MarbleNet 0.850. **CadenceVAD does not reach these.**

The full table, including every ablation, is in
[`LOCAL_RESULTS.md`](LOCAL_RESULTS.md); machine-readable reports with per-clip
AUCs and provenance are in `benchmarks/ava-speech/`.

## Statistical power

Five identical runs differing only in seed give sd 0.0127 on this benchmark, while
their development-split F1 varies by sd 0.0008. **Effects below roughly 0.02 AUC
are not resolvable at three seeds.** Any comparison in this repository smaller
than that should be read as unresolved, not as a result. See
[`LOCAL_RESULTS.md`](LOCAL_RESULTS.md#reproducibility-floor).
