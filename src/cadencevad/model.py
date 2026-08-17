from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig
from .modulation import CausalModulationFilterbank


class CausalDepthwiseBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cache_size = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def _post(self, residual: Tensor, encoded: Tensor) -> Tensor:
        encoded = encoded.transpose(1, 2)
        encoded = self.norm(encoded)
        encoded = F.silu(encoded)
        return residual + self.dropout(encoded)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        channel_first = inputs.transpose(1, 2)
        encoded = self.depthwise(F.pad(channel_first, (self.cache_size, 0)))
        encoded = self.pointwise(encoded)
        return self._post(residual, encoded)

    def initial_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(
            batch_size, self.depthwise.in_channels, self.cache_size, device=device, dtype=dtype
        )

    def stream(self, inputs: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        # inputs: [batch, 1, channels]
        residual = inputs
        current = inputs.transpose(1, 2)
        context = torch.cat((cache, current), dim=-1)
        encoded = self.pointwise(self.depthwise(context))
        next_cache = context[..., -self.cache_size :]
        return self._post(residual, encoded), next_cache


class CausalRunningNorm(nn.Module):
    """Causal per-feature mean/variance normalization with a fixed decay.

    Channel, microphone and gain differences show up as slow per-band level
    offsets. A frame-local operation cannot see them, so the v0.1 frontend's
    across-band mean subtraction leaves the model exposed to them. This tracks each
    feature's own running mean and variance instead, which is the classic causal
    CMVN idea and the streaming counterpart of the windowed per-band normalization
    that recent compact VADs credit for their robustness.

    The running statistics start at zero so the module satisfies the repository's
    streaming contract, where every state tensor is zero-initialized by the ONNX
    runtime and the embedded C runtime alike. The variance floor keeps the first
    frames finite while the statistics warm up.

    Output is the raw feature concatenated with its normalized counterpart, so the
    model keeps absolute loudness as evidence while also seeing a channel-invariant
    view. That doubles the input projection but adds no parameters here.
    """

    def __init__(self, feature_dim: int, rate: float, variance_floor: float) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.rate = float(rate)
        self.decay = 1.0 - float(rate)
        self.variance_floor = float(variance_floor)

    @property
    def output_dim(self) -> int:
        return 2 * self.feature_dim

    def initial_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        return torch.zeros(batch_size, 2 * self.feature_dim, device=device, dtype=dtype)

    def _block_size(self) -> int:
        """Longest block whose inverse-decay weights stay well inside float32.

        The scan below divides by ``decay**k``. Capping that growth at about 1e6
        keeps the cumulative sum accurate while still covering whole training
        chunks in a handful of blocks.
        """
        if self.decay >= 1.0:
            return 256
        import math

        limit = int(math.log(1e6) / -math.log(self.decay)) + 1
        return max(16, min(256, limit))

    def _scan(self, values: Tensor, initial: Tensor) -> Tensor:
        """Exponential moving average along time, exactly and in parallel.

        For a block, ``m_t = decay**(t+1) * m_-1 + rate * sum_k decay**(t-k) x_k``.
        Factoring out ``decay**t`` turns the inner sum into a cumulative sum, so a
        whole block resolves without stepping frame by frame. Blocks are sized so
        the factored-out weights never leave float32's accurate range.
        """
        batch, frames, dim = values.shape
        block = self._block_size()
        outputs = []
        state = initial
        for start in range(0, frames, block):
            chunk = values[:, start : start + block]
            length = chunk.shape[1]
            steps = torch.arange(length, device=values.device, dtype=values.dtype)
            decay = torch.as_tensor(self.decay, device=values.device, dtype=values.dtype)
            forward = decay.pow(steps).view(1, length, 1)
            inverse = decay.pow(-steps).view(1, length, 1)
            cumulative = torch.cumsum(chunk * inverse, dim=1)
            averaged = forward * (decay * state.unsqueeze(1) + self.rate * cumulative)
            outputs.append(averaged)
            state = averaged[:, -1]
        return torch.cat(outputs, dim=1)

    def forward(self, features: Tensor, state: Tensor | None = None) -> Tensor:
        if state is None:
            state = features.new_zeros(features.shape[0], 2 * self.feature_dim)
        mean = self._scan(features, state[:, : self.feature_dim])
        centered = features - mean
        variance = self._scan(centered.square(), state[:, self.feature_dim :])
        normalized = centered * torch.rsqrt(variance + self.variance_floor)
        return torch.cat((features, normalized), dim=-1)

    def stream(self, feature: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Single-frame update, numerically identical to one step of ``_scan``."""
        mean = self.decay * state[:, : self.feature_dim] + self.rate * feature.squeeze(1)
        centered = feature.squeeze(1) - mean
        variance = (
            self.decay * state[:, self.feature_dim :] + self.rate * centered.square()
        )
        normalized = centered * torch.rsqrt(variance + self.variance_floor)
        output = torch.cat((feature.squeeze(1), normalized), dim=-1).unsqueeze(1)
        return output, torch.cat((mean, variance), dim=-1)


@dataclass
class StreamingModelState:
    convolution: tuple[Tensor, ...]
    recurrent: Tensor
    input_norm: Tensor | None = None
    modulation: Tensor | None = None


class CadenceVad(nn.Module):
    """Small causal frame model.

    The auxiliary three-way head represents speech, music, and other vocal
    events. Binary training can ignore it; richer datasets can supervise it.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        self.modulation: CausalModulationFilterbank | None = None
        projection_dim = self.config.feature_dim
        if self.config.modulation:
            self.modulation = CausalModulationFilterbank(
                num_bands=self.config.modulation_bands,
                num_filters=self.config.modulation_filters,
                num_groups=self.config.modulation_groups,
            )
            projection_dim += self.modulation.output_dim
        self.running_norm: CausalRunningNorm | None = None
        if self.config.input_norm == "ema":
            # Normalizes whatever reaches it, modulation features included.
            self.running_norm = CausalRunningNorm(
                projection_dim,
                self.config.input_norm_rate,
                self.config.input_norm_variance_floor,
            )
            projection_dim = self.running_norm.output_dim
        self.input_norm = nn.LayerNorm(projection_dim)
        self.input_projection = nn.Linear(projection_dim, self.config.hidden_dim)
        self.blocks = nn.ModuleList(
            CausalDepthwiseBlock(
                self.config.hidden_dim,
                self.config.kernel_size,
                dilation,
                self.config.dropout,
            )
            for dilation in self.config.dilations
        )
        self.recurrent = nn.GRU(
            self.config.hidden_dim,
            self.config.recurrent_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.config.recurrent_dim)
        self.speech_head = nn.Linear(self.config.recurrent_dim, 1)
        self.auxiliary_head = nn.Linear(self.config.recurrent_dim, 3)

    def _input(self, features: Tensor) -> Tensor:
        return F.silu(self.input_projection(self.input_norm(features)))

    def _heads(self, encoded: Tensor) -> dict[str, Tensor]:
        encoded = self.output_norm(encoded)
        return {
            "speech_logits": self.speech_head(encoded).squeeze(-1),
            "auxiliary_logits": self.auxiliary_head(encoded),
        }

    def _extend(self, features: Tensor) -> Tensor:
        """Append modulation-filterbank energies to the frame features."""
        if self.modulation is None:
            return features
        bands = features[..., : self.config.modulation_bands]
        energies, _ = self.modulation(bands)
        return torch.cat((features, energies), dim=-1)

    def encode(self, features: Tensor) -> Tensor:
        """Everything ahead of the recurrence, as one step.

        Callers that need to drive the GRU themselves - a long-sequence evaluator
        chunking it to stay inside cuDNN's limits, for instance - must go through
        this rather than reassembling the preamble, so a new input stage cannot be
        silently skipped.
        """
        features = self._extend(features)
        if self.running_norm is not None:
            features = self.running_norm(features)
        encoded = self._input(features)
        for block in self.blocks:
            encoded = block(encoded)
        return encoded

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        encoded = self.encode(features)
        encoded, _ = self.recurrent(encoded)
        return self._heads(encoded)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> StreamingModelState:
        target = torch.device(device)
        caches = tuple(block.initial_cache(batch_size, target, dtype) for block in self.blocks)
        recurrent = torch.zeros(
            1, batch_size, self.config.recurrent_dim, device=target, dtype=dtype
        )
        running = (
            self.running_norm.initial_cache(batch_size, target, dtype)
            if self.running_norm is not None
            else None
        )
        modulation = (
            self.modulation.initial_cache(batch_size, target, dtype)
            if self.modulation is not None
            else None
        )
        return StreamingModelState(caches, recurrent, running, modulation)

    def stream_step(
        self,
        feature: Tensor,
        state: StreamingModelState,
    ) -> tuple[dict[str, Tensor], StreamingModelState]:
        if feature.ndim == 2:
            feature = feature.unsqueeze(1)
        if feature.ndim != 3 or feature.shape[1] != 1:
            raise ValueError("stream_step expects [batch, features] or [batch, 1, features]")
        modulation = state.modulation
        if self.modulation is not None:
            if modulation is None:
                raise ValueError("streaming state is missing the modulation cache")
            energies, modulation = self.modulation.stream(
                feature[:, 0, : self.config.modulation_bands], modulation
            )
            feature = torch.cat((feature, energies), dim=-1)
        running = state.input_norm
        if self.running_norm is not None:
            if running is None:
                raise ValueError("streaming state is missing the input-norm cache")
            feature, running = self.running_norm.stream(feature, running)
        encoded = self._input(feature)
        next_caches: list[Tensor] = []
        for block, cache in zip(self.blocks, state.convolution, strict=True):
            encoded, next_cache = block.stream(encoded, cache)
            next_caches.append(next_cache)
        encoded, recurrent = self.recurrent(encoded, state.recurrent)
        return self._heads(encoded), StreamingModelState(
            tuple(next_caches), recurrent, running, modulation
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
