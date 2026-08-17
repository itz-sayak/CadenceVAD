# Benchmark artifacts

## `ava-speech/` — primary accuracy benchmark

Full AVA-Speech, 39.667 h (99.2% of the 40 h nominal), strictly causal, 10 ms
frames, no smoothing, no future context. One JSON per model, each carrying its
protocol block, adapter metadata, per-condition AUCs, per-clip AUCs, a
1,000-iteration clip-level bootstrap interval and dataset provenance.

| Model | Params | AUC | 95% CI | FA | Miss |
|---|---:|---:|---|---:|---:|
| Silero VAD | — | 0.9022 | [0.8910, 0.9127] | 0.140 | 0.186 |
| TEN VAD | — | 0.8966 | [0.8905, 0.9026] | 0.177 | 0.180 |
| conv-v3-m-rank | 99,930 | 0.8436 | [0.8242, 0.8494] | 0.164 | 0.270 |
| conv-v1-base | 46,170 | 0.8418 | [0.8300, 0.8536] | 0.157 | 0.292 |
| v0.1 baseline | 46,170 | 0.8295 | [0.8187, 0.8408] | 0.358 | 0.181 |

Reproduce with `scripts/prepare_ava_speech.py` then
`scripts/benchmark_ava_speech.py`; summarise with
`scripts/summarize_ava_results.py`. Protocol: `docs/AVA_SPEECH.md`.

## `portable-runtime/` — latency

Idle AMD Ryzen Threadripper PRO 7965WX, Linux x86-64, single thread, pinned core,
50,000 iterations after 5,000 warm-up. Complete causal frontend plus model for one
10 ms hop.

| Build | p50 | p95 | p99 | RTF |
|---|---:|---:|---:|---:|
| `linux-x86_64-avx2.json` | 9.02 µs | 9.05 µs | 9.83 µs | 0.00090 |
| `linux-x86_64-scalar.json` | 27.06 µs | 28.33 µs | 30.21 µs | 0.00271 |

Build with `cmake -S native/portable -B build -DCADENCEVAD_ENABLE_AVX2=ON` and run
`cadencevad_bench`. Windows builds from the same source with MSVC and is
parity-checked in CI, but no Windows timings are recorded — no Windows host was
available and estimating them would be misleading.

## `study/` — controlled experiments

JSONL, one row per training run, each with its AVA result, development metrics and
manifest.

- `seed-variance.jsonl`: five seeds, identical config and data. mean 0.8069,
  **sd 0.0127**, range 0.0302. This is the noise floor every other comparison must
  clear.
- `seed-variance-v2.jsonl`: a failed attempt to reduce that variance with 200
  epochs and Polyak averaging. mean 0.7101, sd 0.0322 — worse on both counts, and
  confounded with a dataset change.
- `clmf-ablation.jsonl`: modulation filterbank, three seeds per arm on matched
  data. +0.005 overall (0.3 SE), +0.016 on music (1.3 SE). Not significant.

## `cadencevad-v0.1/` — baseline artifacts

Retained evidence for the third-party v0.1 baseline checkpoint, including its
exploratory TEN public-set evaluation and Apple M4 Pro runtime measurements. These
describe that baseline, not a CadenceVAD-trained model.

The external-model harness is `scripts/benchmark_official_vads.py`. Its retained
artifact records exact source revisions, artifact hashes, machine/runtime
metadata, and a declared protocol. Its compute-advantage field uses
audio-normalized real-time factor because Silero's hop is 32 ms while the other
measured paths use 10 ms.

To reconstruct the descriptive TEN manifest from an independently obtained
checkout of the recorded revision:

```bash
python scripts/convert_scv_manifest.py \
  --input /path/to/ten-vad/testset \
  --output data/ten-public/manifest.jsonl \
  --relative-paths \
  --condition ten-public-testset \
  --expected-items 30 \
  --source-repository https://github.com/TEN-framework/ten-vad \
  --source-revision 22a3bcd4509d0faaa8eef4881e8af5f39c178950
```

The converter verifies mono 16 kHz PCM16 input and writes file hashes plus a
manifest digest to `manifest.provenance.json`. TEN's repository has additional
Agora licence conditions, and the test-set README does not grant a separate
redistribution licence for the constituent audio. Do not vendor the test set;
review its terms before use.

All retained timings are warm local CadenceVAD measurements on the Apple M4 Pro
described in `docs/LOCAL_RESULTS.md`. Different scopes mean raw call latency is
not an accuracy, endpoint-latency, or product ranking.
