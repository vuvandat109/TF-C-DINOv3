"""CPU-only tests: python test_dinov3_loss.py (no HF access required)."""

import torch
from dinov3_loss import DINOv3TFCLoss


def test_loss_and_gradients():
    torch.manual_seed(42)
    criterion = DINOv3TFCLoss(temperature=0.2)
    batch_size = 5
    embeddings = [torch.randn(batch_size, 128, requires_grad=True) for _ in range(4)]
    loss, parts = criterion(*embeddings)
    assert loss.ndim == 0 and torch.isfinite(loss), "loss must be finite scalar"
    assert set(parts) == {"time", "frequency", "cross_time_frequency"}
    assert all(torch.isfinite(x) for x in parts.values())
    loss.backward()
    for i, x in enumerate(embeddings):
        assert x.grad is not None and torch.isfinite(x.grad).all(), f"No gradient in z[{i}]"
        assert x.grad.abs().sum() > 0, f"Zero gradient in z[{i}]"


def test_two_sample_batch():
    criterion = DINOv3TFCLoss()
    inputs = [torch.randn(2, 128, requires_grad=True) for _ in range(4)]
    result, _ = criterion(*inputs)
    result.backward()
    assert torch.isfinite(result)


def test_invalid_batch():
    criterion = DINOv3TFCLoss()
    try:
        criterion(*[torch.randn(1, 128) for _ in range(4)])
    except ValueError as error:
        assert "at least 2 samples" in str(error)
    else:
        raise AssertionError("batch size 1 must be rejected")


def test_positive_pairs_better_than_shuffle():
    torch.manual_seed(7)
    criterion = DINOv3TFCLoss(temperature=0.1)
    anchor = torch.randn(8, 128)
    aligned, _ = criterion(anchor, anchor, anchor, anchor)
    shuffled, _ = criterion(anchor, anchor.roll(1, 0), anchor.roll(2, 0), anchor.roll(3, 0))
    assert aligned < shuffled, (aligned, shuffled)


def test_frozen_backbone_trainable_projectors():
    # Mimics the trainable output path of real TF-C with DINOv3 frozen.
    from torch import nn
    torch.manual_seed(123)
    frozen_t, frozen_f = nn.Linear(20, 32), nn.Linear(20, 32)
    frozen_t.requires_grad_(False)
    frozen_f.requires_grad_(False)
    projector_t, projector_f = nn.Linear(32, 128), nn.Linear(32, 128)
    source = torch.randn(4, 20)
    altered = source + 0.2 * torch.randn_like(source)
    z_t, z_t_aug = projector_t(frozen_t(source)), projector_t(frozen_t(altered))
    z_f, z_f_aug = projector_f(frozen_f(source)), projector_f(frozen_f(altered))
    loss, _ = DINOv3TFCLoss()(z_t, z_t_aug, z_f, z_f_aug)
    loss.backward()
    assert projector_t.weight.grad is not None and projector_t.weight.grad.norm() > 0
    assert projector_f.weight.grad is not None and projector_f.weight.grad.norm() > 0
    assert frozen_t.weight.grad is None and frozen_f.weight.grad is None


if __name__ == "__main__":
    test_loss_and_gradients()
    test_two_sample_batch()
    test_invalid_batch()
    test_positive_pairs_better_than_shuffle()
    test_frozen_backbone_trainable_projectors()
    print("PASS: DINOv3-TF-C loss, positive pairing and gradients in all four projected views")
