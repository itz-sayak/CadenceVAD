from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16_000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 40
    f_min: float = 50.0
    f_max: float = 7_600.0
    # Subtract each frame's mean across mel bands. This normalizes spectral tilt
    # but discards per-band level, so it is disabled when the model applies its own
    # causal per-band normalization instead. True reproduces the v0.1 frontend.
    mel_mean_subtraction: bool = True

    @property
    def frame_samples(self) -> int:
        return round(self.sample_rate * self.frame_ms / 1_000)

    @property
    def hop_samples(self) -> int:
        return round(self.sample_rate * self.hop_ms / 1_000)

    @property
    def feature_dim(self) -> int:
        return self.n_mels + 3

    def validate(self) -> None:
        if self.sample_rate not in (8_000, 16_000):
            raise ValueError("sample_rate must be 8000 or 16000")
        if self.frame_samples <= self.hop_samples:
            raise ValueError("frame must be longer than hop")
        if self.n_fft < self.frame_samples:
            raise ValueError("n_fft must be at least frame_samples")
        if not 0 <= self.f_min < self.f_max <= self.sample_rate / 2:
            raise ValueError("mel frequency bounds are invalid")


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int = 43
    hidden_dim: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4)
    recurrent_dim: int = 56
    dropout: float = 0.08
    # "layer" reproduces the v0.1 input stage. "ema" prepends a causal per-band
    # running mean/variance normalizer and feeds the model both the raw and the
    # normalized view, which makes it robust to channel and level drift without
    # giving up absolute loudness as evidence.
    input_norm: str = "layer"
    # EMA smoothing rate per frame. 0.01 at a 10 ms hop is roughly a 1 s window.
    input_norm_rate: float = 0.01
    # Floor on the running variance, in squared log-feature units. Stops the
    # normalizer amplifying near-constant bands into noise.
    input_norm_variance_floor: float = 1.0
    # Learnable causal modulation filterbank over the mel trajectory. Speech
    # modulates at the syllabic rate while music sustains, so these give the model
    # an explicit speech/music cue it otherwise has to infer. Off reproduces v0.1.
    modulation: bool = False
    modulation_bands: int = 40
    modulation_filters: int = 4
    modulation_groups: int = 5

    def validate(self) -> None:
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if any(dilation < 1 for dilation in self.dilations):
            raise ValueError("dilations must be positive")
        if self.hidden_dim < 8 or self.recurrent_dim < 8:
            raise ValueError("model dimensions are too small")
        if self.input_norm not in ("layer", "ema"):
            raise ValueError("input_norm must be 'layer' or 'ema'")
        if not 0.0 < self.input_norm_rate <= 1.0:
            raise ValueError("input_norm_rate must be in (0, 1]")
        if self.input_norm_variance_floor <= 0.0:
            raise ValueError("input_norm_variance_floor must be positive")
        if self.modulation:
            if not 1 <= self.modulation_bands <= self.feature_dim:
                raise ValueError("modulation_bands must fit inside feature_dim")
            if self.modulation_filters < 1 or self.modulation_groups < 1:
                raise ValueError("modulation filter and group counts must be positive")
            if self.modulation_groups > self.modulation_bands:
                raise ValueError("modulation_groups cannot exceed modulation_bands")


@dataclass(frozen=True)
class TrainingConfig:
    chunk_seconds: float = 4.0
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    positive_weight: float = 1.4
    focal_gamma: float = 0.0
    boundary_weight: float = 0.15
    auxiliary_weight: float = 0.10
    gradient_clip: float = 5.0
    detector_max_false_alarm_rate: float = 0.2
    # Weight on the pairwise ranking term. ROC-AUC is a ranking metric, so this
    # optimizes it directly alongside cross entropy. Zero reproduces v0.1.
    ranking_weight: float = 0.0
    ranking_margin: float = 1.0
    ranking_pairs: int = 4_096
    # Polyak averaging of the weights. Selecting the best-dev-F1 epoch is unstable
    # when the development split is saturated: it varies by far less than the
    # resulting cross-domain accuracy does, so the choice is close to arbitrary.
    # Averaging removes that lottery and gives a deterministic final model.
    # "none" reproduces v0.1 best-epoch selection.
    weight_averaging: str = "none"
    weight_averaging_decay: float = 0.999


@dataclass(frozen=True)
class DetectorConfig:
    start_threshold: float = 0.62
    stop_threshold: float = 0.36
    start_frames: int = 2
    stop_frames: int = 10
    pre_roll_frames: int = 3

    def validate(self) -> None:
        if not 0 < self.stop_threshold < self.start_threshold < 1:
            raise ValueError("expected 0 < stop_threshold < start_threshold < 1")
        if self.start_frames < 1 or self.stop_frames < 1 or self.pre_roll_frames < 0:
            raise ValueError("frame counts must be non-negative")


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 1337
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def validate(self) -> None:
        self.feature.validate()
        self.model.validate()
        self.detector.validate()
        if not 0.0 <= self.training.detector_max_false_alarm_rate <= 1.0:
            raise ValueError("detector_max_false_alarm_rate must be between zero and one")
        if self.training.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative")
        if self.training.ranking_weight < 0.0:
            raise ValueError("ranking_weight must be non-negative")
        if self.training.ranking_margin <= 0.0:
            raise ValueError("ranking_margin must be positive")
        if self.training.ranking_pairs < 1:
            raise ValueError("ranking_pairs must be at least one")
        if self.training.weight_averaging not in ("none", "ema"):
            raise ValueError("weight_averaging must be 'none' or 'ema'")
        if not 0.0 < self.training.weight_averaging_decay < 1.0:
            raise ValueError("weight_averaging_decay must be in (0, 1)")
        if self.model.feature_dim != self.feature.feature_dim:
            raise ValueError(
                f"model feature_dim={self.model.feature_dim} does not match "
                f"frontend feature_dim={self.feature.feature_dim}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProjectConfig:
        model_raw = dict(raw.get("model", {}))
        if "dilations" in model_raw:
            model_raw["dilations"] = tuple(model_raw["dilations"])
        config = cls(
            seed=int(raw.get("seed", 1337)),
            feature=FeatureConfig(**raw.get("feature", {})),
            model=ModelConfig(**model_raw),
            training=TrainingConfig(**raw.get("training", {})),
            detector=DetectorConfig(**raw.get("detector", {})),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> ProjectConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
