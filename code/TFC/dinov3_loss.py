"""Contrastive loss for TF-C with two independent DINOv3 backbones.

All contrastive terms operate on the trainable 128-D projectors, NOT on
frozen DINOv3 backbone features. This makes the loss train the new heads.
"""

import torch
import torch.nn.functional as F
from torch import nn


class DINOv3TFCLoss(nn.Module):
    """Two intra-view NT-Xent terms plus symmetric time/frequency alignment.

    Args:
        temperature: contrastive temperature (>0).
        time_weight: temporal augmentation term coefficient.
        frequency_weight: spectral augmentation term coefficient.
        cross_weight: time-frequency alignment coefficient.

    Inputs are [batch, embedding_dim] projected embeddings:
        z_t: original temporal view, z_t_aug: augmented temporal view,
        z_f: original spectral view, z_f_aug: augmented spectral view.

    For each cross-modal pairing the positive is the same input sample;
    other samples in the batch are negatives. Requires batch_size >= 2.
    """

    def __init__(self, temperature=0.2, time_weight=0.2,
                 frequency_weight=0.2, cross_weight=1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if min(time_weight, frequency_weight, cross_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if not any((time_weight, frequency_weight, cross_weight)):
            raise ValueError("at least one loss weight must be positive")
        self.temperature = float(temperature)
        self.time_weight = float(time_weight)
        self.frequency_weight = float(frequency_weight)
        self.cross_weight = float(cross_weight)

    def _nt_xent(self, a, b):
        if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
            raise ValueError("NT-Xent inputs must be [B,D] with matching shapes")
        batch_size = a.shape[0]
        if batch_size < 2:
            raise ValueError("Contrastive training needs at least 2 samples per batch")
        z = F.normalize(torch.cat((a, b), dim=0), p=2, dim=1)
        logits = torch.matmul(z, z.T) / self.temperature
        n = 2 * batch_size
        logits = logits.masked_fill(torch.eye(n, device=z.device, dtype=torch.bool),
                                    torch.finfo(logits.dtype).min)
        target = (torch.arange(n, device=z.device) + batch_size) % n
        return F.cross_entropy(logits, target)

    def forward(self, z_t, z_t_aug, z_f, z_f_aug):
        if z_t.shape != z_t_aug.shape or z_t.shape != z_f.shape or z_t.shape != z_f_aug.shape:
            raise ValueError("All four projected embeddings must match in shape")
        loss_time = self._nt_xent(z_t, z_t_aug)
        loss_frequency = self._nt_xent(z_f, z_f_aug)
        loss_cross = torch.stack((
            self._nt_xent(z_t, z_f),
            self._nt_xent(z_t, z_f_aug),
            self._nt_xent(z_t_aug, z_f),
            self._nt_xent(z_t_aug, z_f_aug),
        )).mean()
        total = (self.time_weight * loss_time
                 + self.frequency_weight * loss_frequency
                 + self.cross_weight * loss_cross)
        parts = {"time": loss_time.detach(),
                 "frequency": loss_frequency.detach(),
                 "cross_time_frequency": loss_cross.detach()}
        return total, parts
