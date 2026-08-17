import numpy as np
import pytest
import torch

from cadencevad.config import FeatureConfig, ModelConfig, ProjectConfig
from cadencevad.features import CausalFeatureExtractor
from cadencevad.model import CadenceVad, CausalRunningNorm


def _naive_running_norm(
    features: torch.Tensor, rate: float, floor: float
) -> torch.Tensor:
    """Reference implementation written as a plain frame-by-frame recurrence."""
    decay = 1.0 - rate
    batch, frames, dim = features.shape
    mean = torch.zeros(batch, dim, dtype=features.dtype)
    variance = torch.zeros(batch, dim, dtype=features.dtype)
    outputs = []
    for index in range(frames):
        current = features[:, index]
        mean = decay * mean + rate * current
        centered = current - mean
        variance = decay * variance + rate * centered.square()
        outputs.append(
            torch.cat((current, centered * torch.rsqrt(variance + floor)), dim=-1)
        )
    return torch.stack(outputs, dim=1)


@pytest.mark.parametrize("rate", [0.5, 0.1, 0.01, 0.002])
def test_block_scan_matches_frame_recurrence(rate: float) -> None:
    """The parallel scan must be the same computation as stepping frame by frame."""
    torch.manual_seed(11)
    module = CausalRunningNorm(43, rate, 1.0)
    features = torch.randn(3, 977, 43) * 4.0 - 2.0

    assert torch.allclose(
        module(features),
        _naive_running_norm(features, rate, 1.0),
        atol=1e-5,
    )


def test_streaming_state_matches_offline_scan() -> None:
    torch.manual_seed(12)
    module = CausalRunningNorm(43, 0.01, 1.0)
    features = torch.randn(2, 500, 43)

    offline = module(features)
    state = module.initial_cache(2, torch.device("cpu"), torch.float32)
    steps = []
    for index in range(features.shape[1]):
        output, state = module.stream(features[:, index : index + 1], state)
        steps.append(output)

    assert torch.allclose(offline, torch.cat(steps, dim=1), atol=1e-5)


def test_running_norm_state_is_zero_initialised() -> None:
    """The ONNX and embedded runtimes zero every state tensor, so zeros must be valid."""
    module = CausalRunningNorm(8, 0.01, 1.0)
    cache = module.initial_cache(2, torch.device("cpu"), torch.float32)

    assert torch.count_nonzero(cache) == 0
    output, _ = module.stream(torch.randn(2, 1, 8), cache)
    assert torch.isfinite(output).all()


def test_running_norm_is_invariant_to_a_constant_band_offset() -> None:
    """A per-band level shift is what channel change looks like; it must wash out."""
    torch.manual_seed(13)
    module = CausalRunningNorm(16, 0.05, 1e-4)
    features = torch.randn(1, 4_000, 16)
    offset = torch.randn(1, 1, 16) * 5.0

    plain = module(features)[:, 2_000:, 16:]
    shifted = module(features + offset)[:, 2_000:, 16:]

    assert torch.allclose(plain, shifted, atol=1e-2)


def test_ema_model_streaming_matches_offline() -> None:
    torch.manual_seed(14)
    config = ModelConfig(dilations=(1, 2, 4, 8), recurrent_dim=64, input_norm="ema")
    model = CadenceVad(config).eval()
    features = torch.randn(2, 400, config.feature_dim)

    with torch.inference_mode():
        offline = torch.sigmoid(model(features)["speech_logits"])
        state = model.initial_state(2, "cpu")
        steps = []
        for index in range(features.shape[1]):
            head, state = model.stream_step(features[:, index : index + 1], state)
            steps.append(torch.sigmoid(head["speech_logits"]))

    assert torch.allclose(offline, torch.cat(steps, dim=1), atol=1e-5)


def test_ema_model_keeps_the_parameter_budget() -> None:
    base = CadenceVad(ModelConfig(dilations=(1, 2, 4, 8), recurrent_dim=64))
    ema = CadenceVad(
        ModelConfig(dilations=(1, 2, 4, 8), recurrent_dim=64, input_norm="ema")
    )

    assert base.parameter_count == 46_170
    # Only the widened input projection and its norm are added.
    assert ema.parameter_count - base.parameter_count == 2_838
    assert ema.parameter_count < 50_000


def test_default_model_is_unchanged_by_the_new_option() -> None:
    """v0.1 checkpoints must keep loading, so the defaults cannot move."""
    config = ModelConfig()

    assert config.input_norm == "layer"
    assert CadenceVad(config).running_norm is None
    assert FeatureConfig().mel_mean_subtraction is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_norm": "pcen"},
        {"input_norm_rate": 0.0},
        {"input_norm_rate": 1.5},
        {"input_norm_variance_floor": 0.0},
    ],
)
def test_invalid_running_norm_configuration_is_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**overrides).validate()


def test_mel_mean_subtraction_toggle_changes_only_the_mel_bands() -> None:
    torch.manual_seed(15)
    audio = torch.randn(1, 16_000) * 0.1

    with_subtraction = CausalFeatureExtractor(FeatureConfig())(audio)
    without = CausalFeatureExtractor(FeatureConfig(mel_mean_subtraction=False))(audio)

    # The three robustness features sit after the mel bands and are untouched.
    assert torch.allclose(with_subtraction[..., 40:], without[..., 40:], atol=1e-6)
    assert not torch.allclose(with_subtraction[..., :40], without[..., :40], atol=1e-3)
    # Disabling the subtraction is exactly removing the per-frame band mean.
    recovered = without[..., :40] - without[..., :40].mean(dim=-1, keepdim=True)
    assert torch.allclose(with_subtraction[..., :40], recovered, atol=1e-5)


def test_numpy_and_torch_frontends_agree_without_mean_subtraction() -> None:
    """The ONNX path uses the NumPy frontend, so the toggle must land in both."""
    from cadencevad.numpy_features import (
        NumpyCausalFeatureExtractor,
        NumpyStreamingFeatureExtractor,
    )

    config = FeatureConfig(mel_mean_subtraction=False)
    audio = np.random.default_rng(16).normal(0.0, 0.1, 8_000).astype(np.float32)

    torch_features = CausalFeatureExtractor(config)(torch.from_numpy(audio)).numpy()
    streaming = NumpyStreamingFeatureExtractor(NumpyCausalFeatureExtractor(config))
    numpy_features = streaming.push(audio)

    usable = min(torch_features.shape[-2], numpy_features.shape[0])
    assert np.abs(torch_features[0, :usable] - numpy_features[:usable]).max() < 1e-3


def test_project_config_round_trips_the_new_fields() -> None:
    config = ProjectConfig(
        feature=FeatureConfig(mel_mean_subtraction=False),
        model=ModelConfig(dilations=(1, 2, 4, 8), recurrent_dim=64, input_norm="ema"),
    )

    restored = ProjectConfig.from_dict(config.to_dict())

    assert restored.model.input_norm == "ema"
    assert restored.feature.mel_mean_subtraction is False
