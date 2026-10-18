"""TF-C -> DINOv3 differentiable signal-to-image adapters.

Two distinct views: waveform/derivative/envelope for the temporal branch;
log-STFT magnitude for the frequency branch. No labels are used.
"""
import torch
from torch import nn
from torch.nn import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SignalImageAdapter(nn.Module):
    def __init__(self, domain: str, image_size: int = 224, n_fft: int = 64):
        super().__init__()
        if domain not in ("time", "frequency"):
            raise ValueError("domain must be 'time' or 'frequency'")
        if image_size % 16:
            raise ValueError("image_size must be divisible by 16 for DINOv3 ViT-S/16")
        self.domain, self.image_size, self.n_fft = domain, image_size, n_fft
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, signal):
        # signal [batch, channels, timestamps]. Aggregate multichannel samples
        # to a single waveform, matching the first-channel setting in original TF-C.
        if signal.ndim != 3:
            raise ValueError(f"Expected [B,C,L], got {tuple(signal.shape)}")
        x = signal.float().mean(dim=1, keepdim=True)
        if x.shape[-1] < 4:
            raise ValueError("Signal must contain at least 4 time steps")
        if self.domain == "time":
            waveform = x
            slope = F.pad(x[..., 1:] - x[..., :-1], (1, 0))
            smooth_abs = F.avg_pool1d(x.abs(), kernel_size=5, stride=1, padding=2)
            # Each channel is a different signal descriptor, rendered as a 2D map.
            view = torch.cat([waveform, slope, smooth_abs], dim=1).unsqueeze(2)
        else:
            n_fft = min(self.n_fft, x.shape[-1])
            n_fft = max(4, 2 * (n_fft // 2))
            hop_length = max(1, n_fft // 4)
            window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
            spec = torch.stft(x[:, 0], n_fft=n_fft, hop_length=hop_length,
                              win_length=n_fft, window=window, return_complex=True,
                              center=True).abs()
            log_spec = torch.log1p(spec).unsqueeze(1)
            view = log_spec.repeat(1, 3, 1, 1)
        # Normalize each sample/channel to [0,1]; then use DINOv3 ImageNet stats.
        vmin = view.amin(dim=(-2, -1), keepdim=True)
        vmax = view.amax(dim=(-2, -1), keepdim=True)
        view = (view - vmin) / (vmax - vmin).clamp_min(1e-6)
        view = F.interpolate(view, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)
        return (view - self.mean) / self.std
