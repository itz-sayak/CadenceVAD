"""Causal learnable modulation filterbank.

Speech energy modulates at the syllabic rate, roughly 4-8 Hz across languages.
Music modulates more slowly and sustains notes, and stationary noise barely
modulates at all. That difference is the classic speech/music cue, and it is
exactly where this repository's compact model is weakest: on AVA-Speech its
``SPEECH_WITH_MUSIC`` AUC trails its clean-speech AUC by a wide margin.

A model seeing only instantaneous spectral features plus a few hundred
milliseconds of convolution has to infer modulation structure implicitly. This
module supplies it: a small bank of learnable resonators running along the *time*
axis of every mel band, whose output energies are appended to the frame features.

Fixed 4 Hz modulation tests are known to confuse bass instruments with speech,
which is the argument for *learning* which (spectral band, modulation rate) pairs
discriminate rather than hard-coding one rate.

Three properties make this usable in the streaming runtimes here.

**Zero-initialised state is exactly correct.** An IIR filter started from zero is
the exact response to a signal that was silent beforehand. Every state tensor in
the ONNX and embedded C runtimes is zero-initialised, so no warm-up counter and no
first-frame special case is needed. This is precisely the property PCEN lacks -
its gain smoother conventionally initialises to the first frame's energy - and it
is why this design was chosen over PCEN.

**Stability holds by construction.** The pole radius is sigmoid-bounded strictly
inside the unit circle, so no gradient step can produce a divergent filter and no
projection is required.

**Training stays fast.** A one-pole complex resonator is a first-order recurrence,
so a whole training chunk resolves through the same block scan used by
:class:`cadencevad.model.CausalRunningNorm` instead of a Python loop over frames.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# Modulation rates worth resolving at a 100 Hz frame rate: below the slowest
# phrase rhythm, above the fastest phonetic detail. The syllabic 4-8 Hz band sits
# well inside, but is learned rather than assumed.
MIN_CENTRE_HZ = 0.3
MAX_CENTRE_HZ = 20.0
# Pole radius sets the bandwidth. The lower bound keeps the parallel block scan
# numerically well conditioned; 0.8 is already a ~6 Hz bandwidth, which is wide
# for modulation analysis.
MIN_RADIUS = 0.80
MAX_RADIUS = 0.999
# Leading coefficient of the DC blocker that precedes the bank.
DC_BLOCKER_POLE = 0.995


def _inverse_sigmoid(value: float, low: float, high: float) -> float:
    """Raw parameter whose bounded sigmoid equals ``value``."""
    normalized = min(max((value - low) / (high - low), 1e-4), 1.0 - 1e-4)
    return math.log(normalized / (1.0 - normalized))


def complex_scan(
    values: Tensor,
    pole: Tensor,
    initial: Tensor,
    block: int,
) -> Tensor:
    """``z[t] = pole * z[t-1] + values[t]`` along dim 1, resolved a block at a time.

    Factoring ``pole**t`` out of the running sum turns each block into a cumulative
    sum, so a chunk costs a handful of block steps rather than thousands of
    sequential ones. ``block`` must be small enough that ``|pole|**-block`` stays
    inside floating-point range; :func:`stable_block_size` picks it.

    ``pole`` must already broadcast against ``values``, with a singleton on the
    time axis - for ``[batch, frames, bands, filters]`` that means
    ``[1, 1, 1, filters]``. ``initial`` is ``values`` without its time axis.
    """
    frames = values.shape[1]
    view = (1, -1) + (1,) * (values.ndim - 2)
    outputs = []
    state = initial
    for start in range(0, frames, block):
        chunk = values[:, start : start + block]
        length = chunk.shape[1]
        steps = torch.arange(length, device=values.device, dtype=torch.float32)
        forward = torch.pow(pole, steps.reshape(view))
        inverse = torch.pow(pole, -steps.reshape(view))
        cumulative = torch.cumsum(chunk * inverse, dim=1)
        resolved = forward * (pole * state.unsqueeze(1) + cumulative)
        outputs.append(resolved)
        state = resolved[:, -1]
    return torch.cat(outputs, dim=1)


def stable_block_size(radius: float, headroom: float = 1e6, limit: int = 256) -> int:
    """Longest block whose inverse-pole weights stay within ``headroom``."""
    if radius >= 1.0:
        return limit
    return max(8, min(limit, int(math.log(headroom) / -math.log(radius)) + 1))


class CausalModulationFilterbank(nn.Module):
    """Learnable resonators along the temporal envelope of each mel band.

    Output width is ``num_groups * num_filters``: the per-band modulation energies
    are pooled into contiguous mel-band groups so the feature vector stays small.
    """

    def __init__(
        self,
        num_bands: int = 40,
        num_filters: int = 4,
        num_groups: int = 5,
        frame_rate: float = 100.0,
    ) -> None:
        super().__init__()
        if num_bands < 1 or num_filters < 1 or num_groups < 1:
            raise ValueError("band, filter and group counts must be positive")
        if num_groups > num_bands:
            raise ValueError("num_groups cannot exceed num_bands")
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")

        self.num_bands = num_bands
        self.num_filters = num_filters
        self.num_groups = num_groups
        self.frame_rate = float(frame_rate)

        centres = torch.logspace(
            math.log10(1.5), math.log10(12.0), num_filters, dtype=torch.float32
        )
        self.centre_raw = nn.Parameter(
            torch.tensor(
                [_inverse_sigmoid(float(c), MIN_CENTRE_HZ, MAX_CENTRE_HZ) for c in centres]
            )
        )
        self.radius_raw = nn.Parameter(
            torch.full((num_filters,), _inverse_sigmoid(0.97, MIN_RADIUS, MAX_RADIUS))
        )
        # Leaky integration of resonator power: about a 250 ms window, long enough
        # to average a syllable without erasing the contrast.
        self.smooth_raw = nn.Parameter(
            torch.full((num_filters,), _inverse_sigmoid(0.04, 0.0, 0.2))
        )

        edges = torch.linspace(0, num_bands, num_groups + 1).round().long()
        pooling = torch.zeros(num_groups, num_bands)
        for group in range(num_groups):
            start, end = int(edges[group]), int(edges[group + 1])
            end = max(end, start + 1)
            pooling[group, start:end] = 1.0 / float(end - start)
        self.register_buffer("pooling", pooling)

    @property
    def output_dim(self) -> int:
        return self.num_groups * self.num_filters

    @property
    def state_dim(self) -> int:
        """Per-sample state: DC blocker (2/band) plus resonator (2) and power (1)."""
        return self.num_bands * (2 + self.num_filters * 3)

    def centre_hz(self) -> Tensor:
        return MIN_CENTRE_HZ + (MAX_CENTRE_HZ - MIN_CENTRE_HZ) * torch.sigmoid(
            self.centre_raw
        )

    def radius(self) -> Tensor:
        return MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * torch.sigmoid(self.radius_raw)

    def smoothing(self) -> Tensor:
        return (0.2 * torch.sigmoid(self.smooth_raw)).clamp(1e-3, 0.199)

    def poles(self) -> Tensor:
        """Complex resonator poles; ``|p| < 1`` for every reachable parameter."""
        angle = 2.0 * math.pi * self.centre_hz() / self.frame_rate
        return torch.polar(self.radius(), angle)

    def initial_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        return torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)

    def _split_state(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = state.shape[0]
        bands = self.num_bands
        filters = self.num_filters
        cursor = 0

        def take(width: int, shape: tuple[int, ...]) -> Tensor:
            nonlocal cursor
            piece = state[:, cursor : cursor + width].reshape(shape)
            cursor += width
            return piece

        previous_input = take(bands, (batch, bands))
        previous_output = take(bands, (batch, bands))
        resonator_real = take(bands * filters, (batch, bands, filters))
        resonator_imag = take(bands * filters, (batch, bands, filters))
        power = take(bands * filters, (batch, bands, filters))
        return previous_input, previous_output, resonator_real, resonator_imag, power

    def _pack_state(
        self,
        previous_input: Tensor,
        previous_output: Tensor,
        resonator: Tensor,
        power: Tensor,
    ) -> Tensor:
        return torch.cat(
            (
                previous_input.flatten(start_dim=1),
                previous_output.flatten(start_dim=1),
                resonator.real.flatten(start_dim=1),
                resonator.imag.flatten(start_dim=1),
                power.flatten(start_dim=1),
            ),
            dim=-1,
        )

    def _block_size(self) -> int:
        return stable_block_size(float(self.radius().min().detach()))

    def forward(
        self,
        bands: Tensor,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """``bands``: ``[batch, frames, num_bands]``.

        Returns the pooled log modulation energies ``[batch, frames, output_dim]``
        and the state after the final frame.
        """
        if bands.ndim != 3:
            raise ValueError("bands must be [batch, frames, num_bands]")
        if bands.shape[-1] != self.num_bands:
            raise ValueError(f"expected {self.num_bands} bands, got {bands.shape[-1]}")
        if state is None:
            state = bands.new_zeros(bands.shape[0], self.state_dim)

        previous_input, previous_output, real, imag, power = self._split_state(state)

        # A resonator still responds at DC, so the slowly drifting absolute level
        # of a mel band would leak into every modulation channel. Block it first.
        shifted = torch.cat((previous_input.unsqueeze(1), bands[:, :-1]), dim=1)
        blocked = complex_scan(
            (bands - shifted).to(torch.complex64),
            torch.full((1, 1, 1), DC_BLOCKER_POLE, device=bands.device, dtype=torch.complex64),
            previous_output.to(torch.complex64),
            stable_block_size(DC_BLOCKER_POLE),
        ).real

        pole = self.poles()
        excitation = blocked.unsqueeze(-1).expand(-1, -1, -1, self.num_filters)
        resonated = complex_scan(
            excitation.to(torch.complex64),
            pole.reshape(1, 1, 1, -1),
            torch.complex(real, imag),
            self._block_size(),
        )

        rate = self.smoothing().reshape(1, 1, 1, -1)
        integrated = complex_scan(
            (rate * resonated.abs().square()).to(torch.complex64),
            (1.0 - rate.reshape(1, 1, 1, -1)).to(torch.complex64),
            power.to(torch.complex64),
            stable_block_size(float((1.0 - self.smoothing()).min().detach())),
        ).real

        pooled = torch.einsum("gn,btnk->btgk", self.pooling, integrated)
        features = torch.log(pooled.clamp_min(1e-8)).flatten(start_dim=2)
        final = self._pack_state(
            bands[:, -1],
            blocked[:, -1],
            resonated[:, -1],
            integrated[:, -1],
        )
        return features, final

    def stream(self, band: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Single-frame update, numerically identical to one step of :meth:`forward`."""
        if band.ndim == 3:
            band = band.squeeze(1)
        previous_input, previous_output, real, imag, power = self._split_state(state)

        blocked = band - previous_input + DC_BLOCKER_POLE * previous_output
        resonated = self.poles().reshape(1, 1, -1) * torch.complex(real, imag)
        resonated = resonated + blocked.unsqueeze(-1).to(torch.complex64)
        rate = self.smoothing().reshape(1, 1, -1)
        integrated = (1.0 - rate) * power + rate * resonated.abs().square()

        pooled = torch.einsum("gn,bnk->bgk", self.pooling, integrated)
        features = torch.log(pooled.clamp_min(1e-8)).flatten(start_dim=1).unsqueeze(1)
        return features, self._pack_state(band, blocked, resonated, integrated)


__all__ = ["CausalModulationFilterbank", "complex_scan", "stable_block_size"]
