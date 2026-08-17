# The causal learnable modulation filterbank

Implementation: [`src/cadencevad/modulation.py`](../src/cadencevad/modulation.py).
Tests: [`tests/test_modulation.py`](../tests/test_modulation.py).

## Why

Speech energy modulates at the syllabic rate, roughly 4–8 Hz across languages.
Music modulates more slowly and sustains notes; stationary noise barely modulates
at all. That is the classic speech/music discriminator, and it maps directly onto
where this model is weakest: on AVA-Speech, `SPEECH_WITH_MUSIC` trails
`CLEAN_SPEECH` by about 0.12 AUC.

A model that sees only instantaneous spectral features plus a 310 ms
convolutional receptive field has to infer modulation structure implicitly from
limited data. This block supplies it directly.

Fixed 4 Hz modulation tests are known to confuse bass instruments with speech.
That is the argument for *learning* which (spectral band, modulation rate) pairs
discriminate rather than hard-coding one rate.

## Design

Operating on the 40-band log-mel trajectory at a 100 Hz frame rate:

1. **DC blocker** per band. A resonator still responds at DC, so a band's slowly
   drifting absolute level would leak into every modulation channel.
2. **`K` learnable one-pole complex resonators**, shared across bands. A one-pole
   complex resonator is a bandpass whose peak sits at the pole angle and whose
   bandwidth is set by the pole radius. Parameterised by centre frequency
   `f ∈ (0.3, 20) Hz` and radius `r ∈ (0.80, 0.999)`, both sigmoid-bounded.
3. **Leaky integration** of `|z|²` gives a causal modulation energy per
   (band, filter), with a learnable rate per filter.
4. **Pooling** into `G` contiguous mel-band groups, then a log.

With `K=4, G=5`: **12 parameters**, 20 extra feature dimensions (`feature_dim`
43 → 63), and 560 floats of state. Total model cost is +1,332 parameters
(46,170 → 47,502), almost all of it the widened input projection.

## Three properties that make it deployable

**Zero-initialised state is exactly correct.** An IIR filter started from zero is
the exact response to a signal that was silent beforehand. Every state tensor in
the ONNX runtime and the embedded C runtime is zero-initialised, so no warm-up
counter and no first-frame special case is needed.

This is the property PCEN lacks. PCEN's gain smoother conventionally initialises
to the first frame's energy, which is incompatible with a zero-init streaming
contract, and it is why PCEN was rejected in favour of this design rather than
bodged into place.

**Stability holds by construction.** The pole radius is sigmoid-bounded strictly
inside the unit circle, so no gradient step can produce a divergent filter and no
projection step is required. `test_poles_stay_inside_the_unit_circle` asserts this
at raw parameter values of ±50.

**Training stays fast.** A one-pole complex resonator is a first-order recurrence,
so `z[t] = p·z[t−1] + x[t]` resolves through a block scan — factoring `p**t` out of
the running sum turns each block into a cumulative sum. A 3,000-frame training
chunk costs a handful of block steps instead of thousands of sequential ones.
Block size is chosen from the pole radius so the factored-out weights stay inside
float32's accurate range.

## Verification

| Property | Result |
|---|---|
| block scan vs frame-by-frame recurrence | matches at radii 0.80 / 0.95 / 0.999 |
| offline vs streaming | 3.3e-6 max deviation |
| drift over a 3,000-frame chunk | 1.0e-6 relative, no growth |
| model offline vs streaming, with filterbank | 1.5e-7 |
| composes with running normalisation | 2.1e-7 |

Frequency selectivity, measured as mean response to a pure modulation at each
rate (group 0):

| filter | 1 Hz | 2 Hz | 4 Hz | 8 Hz | 16 Hz |
|---|---:|---:|---:|---:|---:|
| 1.5 Hz | **144.5** | 140.3 | 12.3 | 2.3 | 0.6 |
| 3.0 Hz | 19.5 | **55.7** | 54.5 | 3.2 | 0.6 |
| 6.0 Hz | 4.0 | 5.1 | **16.2** | 15.9 | 0.8 |
| 12.0 Hz | 1.0 | 1.0 | 1.3 | **4.2** | 4.2 |

Each filter peaks at the rate it is tuned to and falls off monotonically.

## Measured effect

Three seeds per arm, matched data and configuration, evaluated on the full
AVA-Speech benchmark:

| Metric | base | + filterbank | Δ | SE | verdict |
|---|---:|---:|---:|---:|---|
| overall AUC | 0.8056 | 0.8103 | +0.005 | 0.017 | not significant |
| **music AUC** | 0.7634 | 0.7797 | **+0.016** | 0.013 | not significant (1.3 SE) |
| clean AUC | 0.8786 | 0.8596 | −0.019 | 0.020 | not significant |

The music effect is in the predicted direction and is the largest of the three,
which is what the design targets. It does not clear significance.

**This is not a demonstrated improvement.** The block is signal-processing correct
and cheap, but its benefit is unproven. With a seed sd of about 0.019 in each arm,
resolving a +0.016 effect at two standard errors needs roughly 12 seeds per arm.

The honest reading is: suggestive on music, null overall, underpowered.

## Open work

- Run the ablation at 10–12 seeds per arm to settle the music effect.
- Stabilise the training recipe first — a seed sd of 0.0127–0.019 is an order of
  magnitude above the ±0.001 that kiloVAD reports over 10 seeds, so the recipe is
  the binding constraint on measuring anything at this scale.
- Port the filterbank to the C runtime. Biquads and leaky integrators are trivial
  there, but the current portable runtime does not implement it, so a filterbank
  model cannot yet use the 9.02 µs path.
