import math

import pytest
import torch

from cadencevad.config import ModelConfig
from cadencevad.model import CadenceVad
from cadencevad.modulation import (
    MAX_RADIUS,
    MIN_RADIUS,
    CausalModulationFilterbank,
    complex_scan,
    stable_block_size,
)


def _naive_scan(values: torch.Tensor, pole: torch.Tensor, initial: torch.Tensor):
    """Reference first-order complex recurrence, one frame at a time."""
    state = initial
    outputs = []
    for index in range(values.shape[1]):
        state = pole * state + values[:, index]
        outputs.append(state)
    return torch.stack(outputs, dim=1)


@pytest.mark.parametrize("radius", [0.80, 0.95, 0.999])
def test_complex_scan_matches_frame_recurrence(radius: float) -> None:
    torch.manual_seed(3)
    pole = torch.polar(
        torch.tensor([radius]), torch.tensor([2.0 * math.pi * 6.0 / 100.0])
    )
    values = torch.randn(2, 700, 1, dtype=torch.complex64)
    initial = torch.zeros(2, 1, dtype=torch.complex64)

    fast = complex_scan(values, pole.reshape(1, 1, -1), initial, stable_block_size(radius))
    slow = _naive_scan(values, pole.reshape(1, -1), initial)

    assert torch.allclose(fast, slow, atol=1e-4)


def test_streaming_matches_offline() -> None:
    torch.manual_seed(4)
    bank = CausalModulationFilterbank()
    bands = torch.randn(2, 600, 40) * 2.0 - 5.0

    offline, final = bank(bands)
    state = bank.initial_cache(2, torch.device("cpu"), torch.float32)
    steps = []
    for index in range(bands.shape[1]):
        output, state = bank.stream(bands[:, index : index + 1], state)
        steps.append(output)

    assert torch.allclose(offline, torch.cat(steps, dim=1), atol=1e-5)
    assert torch.allclose(final, state, rtol=1e-3, atol=1e-2)


def test_state_is_zero_initialised_and_finite() -> None:
    """The ONNX and embedded runtimes zero every state tensor before the first hop."""
    bank = CausalModulationFilterbank()
    cache = bank.initial_cache(2, torch.device("cpu"), torch.float32)

    assert torch.count_nonzero(cache) == 0
    output, _ = bank.stream(torch.randn(2, 1, 40), cache)
    assert torch.isfinite(output).all()


def test_poles_stay_inside_the_unit_circle() -> None:
    """Stability must hold for every reachable parameter value, not just at init."""
    bank = CausalModulationFilterbank()
    with torch.no_grad():
        for extreme in (-50.0, 50.0):
            bank.centre_raw.fill_(extreme)
            bank.radius_raw.fill_(extreme)
            assert torch.all(bank.poles().abs() < 1.0)
            assert torch.all(bank.radius() >= MIN_RADIUS - 1e-6)
            assert torch.all(bank.radius() <= MAX_RADIUS + 1e-6)


def test_filters_are_selective_for_their_centre_rate() -> None:
    """A resonator must respond most strongly near the rate it is tuned to."""
    bank = CausalModulationFilterbank(num_filters=4, num_groups=1)
    centres = bank.centre_hz().detach()
    frames = 3_000

    responses = []
    for centre in centres:
        time = torch.arange(frames, dtype=torch.float32)
        envelope = torch.sin(2.0 * math.pi * float(centre) * time / 100.0)
        bands = envelope.reshape(1, frames, 1).expand(1, frames, 40).contiguous()
        energy, _ = bank(bands)
        responses.append(energy[0, frames // 2 :].mean(dim=0))

    # Row i is the bank's response to a tone at filter i's own centre rate, so the
    # diagonal must dominate its column.
    matrix = torch.stack(responses)
    for index in range(len(centres)):
        assert matrix[index, index] == matrix[:, index].max()


def test_modulation_energy_separates_modulated_from_steady_input() -> None:
    """A steady band must produce far less modulation energy than a fluctuating one."""
    bank = CausalModulationFilterbank(num_groups=1)
    frames = 2_000
    time = torch.arange(frames, dtype=torch.float32)

    steady = torch.full((1, frames, 40), -3.0)
    syllabic = (
        (-3.0 + 2.0 * torch.sin(2.0 * math.pi * 5.0 * time / 100.0))
        .reshape(1, frames, 1)
        .expand(1, frames, 40)
        .contiguous()
    )

    steady_energy, _ = bank(steady)
    syllabic_energy, _ = bank(syllabic)

    assert syllabic_energy[0, 1_000:].max() > steady_energy[0, 1_000:].max() + 3.0


def test_model_with_modulation_streams_identically() -> None:
    torch.manual_seed(5)
    config = ModelConfig(dilations=(1, 2, 4, 8), recurrent_dim=64, modulation=True)
    model = CadenceVad(config).eval()
    features = torch.randn(2, 300, config.feature_dim)

    with torch.inference_mode():
        offline = torch.sigmoid(model(features)["speech_logits"])
        state = model.initial_state(2, "cpu")
        steps = []
        for index in range(features.shape[1]):
            head, state = model.stream_step(features[:, index : index + 1], state)
            steps.append(torch.sigmoid(head["speech_logits"]))

    assert torch.allclose(offline, torch.cat(steps, dim=1), atol=1e-5)


def test_modulation_composes_with_running_norm() -> None:
    torch.manual_seed(6)
    config = ModelConfig(
        dilations=(1, 2, 4, 8), recurrent_dim=64, modulation=True, input_norm="ema"
    )
    model = CadenceVad(config).eval()
    features = torch.randn(2, 250, config.feature_dim)

    with torch.inference_mode():
        offline = torch.sigmoid(model(features)["speech_logits"])
        state = model.initial_state(2, "cpu")
        steps = []
        for index in range(features.shape[1]):
            head, state = model.stream_step(features[:, index : index + 1], state)
            steps.append(torch.sigmoid(head["speech_logits"]))

    assert torch.allclose(offline, torch.cat(steps, dim=1), atol=1e-5)


def test_modulation_parameter_cost_is_small() -> None:
    base = CadenceVad(ModelConfig(dilations=(1, 2, 4, 8), recurrent_dim=64))
    modulated = CadenceVad(
        ModelConfig(dilations=(1, 2, 4, 8), recurrent_dim=64, modulation=True)
    )

    bank = modulated.modulation
    assert bank is not None
    # Twelve filter parameters; the rest is the widened input projection.
    assert sum(p.numel() for p in bank.parameters()) == 12
    assert modulated.parameter_count - base.parameter_count == 1_332


def test_default_model_has_no_modulation() -> None:
    assert ModelConfig().modulation is False
    assert CadenceVad(ModelConfig()).modulation is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"modulation": True, "modulation_bands": 0},
        {"modulation": True, "modulation_bands": 999},
        {"modulation": True, "modulation_filters": 0},
        {"modulation": True, "modulation_groups": 0},
        {"modulation": True, "modulation_bands": 4, "modulation_groups": 8},
    ],
)
def test_invalid_modulation_configuration_is_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**overrides).validate()


def test_encode_covers_every_input_stage() -> None:
    """The evaluation adapter drives the GRU itself and must not skip a stage.

    It calls ``CadenceVad.encode`` rather than reassembling the preamble. This pins
    that ``encode`` plus the recurrence reproduces ``forward`` for every input-stage
    combination, so adding a stage cannot silently break long-sequence evaluation.
    """
    torch.manual_seed(7)
    for overrides in (
        {},
        {"modulation": True},
        {"input_norm": "ema"},
        {"modulation": True, "input_norm": "ema"},
    ):
        config = ModelConfig(dilations=(1, 2, 4), recurrent_dim=56, **overrides)
        model = CadenceVad(config).eval()
        features = torch.randn(2, 120, config.feature_dim)

        with torch.inference_mode():
            expected = model(features)["speech_logits"]
            encoded = model.encode(features)
            recurrent, _ = model.recurrent(encoded)
            actual = model._heads(recurrent)["speech_logits"]

        assert torch.allclose(expected, actual, atol=1e-6), overrides
