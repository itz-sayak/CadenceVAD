from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def binary_focal_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma: float,
    positive_weight: Tensor | None = None,
) -> Tensor:
    """Binary cross entropy that can focus training on hard frames.

    ``gamma=0`` is exactly weighted binary cross entropy. Soft teacher targets
    are supported by computing the probability assigned to the target mixture.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
        reduction="none",
    )
    if gamma == 0:
        return cross_entropy.mean()
    probabilities = torch.sigmoid(logits)
    target_probability = targets * probabilities + (1.0 - targets) * (
        1.0 - probabilities
    )
    return (cross_entropy * (1.0 - target_probability).pow(gamma)).mean()


def pairwise_ranking_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    margin: float = 1.0,
    pairs: int = 4_096,
    threshold: float = 0.5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Squared-hinge penalty on speech frames scored below non-speech frames.

    Cross entropy optimizes how well calibrated each frame is on its own, but the
    headline VAD metric is ROC-AUC, which only cares about the *ordering* of
    speech against non-speech. This adds that objective directly: for sampled
    (speech, non-speech) pairs it penalises the squared shortfall of their score
    difference below ``margin``.

    Enumerating every pair costs |P|x|N|, which is hundreds of millions of terms
    for a batch of long chunks, so a fixed number of pairs is sampled per step.
    The estimate is unbiased and the variance washes out across steps.

    Returns a zero that still carries a gradient path when a batch happens to
    contain only one class, so the training loop needs no special case.
    """
    if margin <= 0.0:
        raise ValueError("margin must be positive")
    if pairs < 1:
        raise ValueError("pairs must be at least one")
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")

    scores = torch.sigmoid(logits).reshape(-1)
    flat_targets = targets.reshape(-1)
    positive_index = torch.nonzero(flat_targets >= threshold, as_tuple=False).reshape(-1)
    negative_index = torch.nonzero(flat_targets < threshold, as_tuple=False).reshape(-1)
    if positive_index.numel() == 0 or negative_index.numel() == 0:
        return scores.sum() * 0.0

    chosen_positive = torch.randint(
        positive_index.numel(),
        (pairs,),
        device=scores.device,
        generator=generator,
    )
    chosen_negative = torch.randint(
        negative_index.numel(),
        (pairs,),
        device=scores.device,
        generator=generator,
    )
    difference = (
        scores[positive_index[chosen_positive]]
        - scores[negative_index[chosen_negative]]
    )
    return torch.clamp(margin - difference, min=0.0).square().mean()
