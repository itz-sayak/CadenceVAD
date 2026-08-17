# macOS native runtime

The native macOS path embeds the trained weights in the binary and uses Apple Accelerate for:

- the 512-point real FFT;
- mel and neural matrix-vector products;
- a stateful causal convolution/GRU implementation.

It has no Python, PyTorch, ONNX Runtime, model-file parsing, or heap allocation in the hot path.
Every call owns its own `CadenceVadState`; immutable weights are shared by the process.

## Build and benchmark

```bash
uv run cadencevad benchmark-native \
  --checkpoint artifacts/india-gcc/best.pt \
  --output artifacts/india-gcc/native
```

This performs three steps:

1. validates that the checkpoint uses the supported 16 kHz/10 ms architecture;
2. exports generated C weight arrays plus SHA-256 metadata;
3. builds a dynamic library and benchmark with `clang -O3 -mcpu=native`.

To export source without building:

```bash
uv run cadencevad export-native \
  --checkpoint artifacts/india-gcc/best.pt \
  --output artifacts/india-gcc/native/generated
```

For CMake consumers:

```bash
cmake -S native/macos -B artifacts/native-cmake \
  -DCADENCEVAD_WEIGHTS_DIR="$PWD/artifacts/india-gcc/native/generated"
cmake --build artifacts/native-cmake --config Release
```

## API contract

Include `cadencevad_native.h` and the generated `cadencevad_weights.h`.

- `cadencevad_init` creates the small Accelerate FFT setup.
- `cadencevad_process_hop` accepts exactly 160 float samples and returns one probability.
- `cadencevad_push` accepts arbitrary chunk sizes and buffers incomplete hops.
- `cadencevad_reset` clears per-call audio, convolution and recurrent state.
- `cadencevad_destroy` releases the FFT setup.

Audio must already be mono float PCM at 16 kHz. Resampling, echo cancellation and channel mixing
belong before the VAD so their cost and policy stay explicit.

## Correctness gate

The macOS integration test exports a checkpoint, compiles the native library, and checks:

- every feature frame against the PyTorch causal frontend;
- every stateful speech probability against PyTorch across a random stream;
- native benchmark execution.

Changing feature geometry, activation equations, recurrent gate order, compiler flags or numeric
precision requires this parity test and a full real-data accuracy rerun.
