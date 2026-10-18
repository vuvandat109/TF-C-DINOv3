"""TF-C with BOTH original Transformer backbones replaced by DINOv3 ViT.

Time and frequency encoders are *independent* pretrained DINOv3 instances.
"""
import torch
from torch import nn
from dinov3_adapter import SignalImageAdapter

DEFAULT_DINOV3 = "facebook/dinov3-vits16-pretrain-lvd1689m"


class TFC(nn.Module):
    def __init__(self, configs, backbone_name=DEFAULT_DINOV3, freeze_backbone=True,
                 image_size=224, n_fft=64):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("Install dependencies: pip install -r requirements-dinov3.txt") from exc
        # Loading TWICE intentionally: each branch has independent model weights.
        # Requires accepting the DINOv3 license and HF access to model weights.
        self.encoder_t = AutoModel.from_pretrained(backbone_name)
        self.encoder_f = AutoModel.from_pretrained(backbone_name)
        self.time_image = SignalImageAdapter("time", image_size, n_fft)
        self.freq_image = SignalImageAdapter("frequency", image_size, n_fft)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for encoder in (self.encoder_t, self.encoder_f):
                encoder.requires_grad_(False)
                encoder.eval()
        dim_t = self.encoder_t.config.hidden_size
        dim_f = self.encoder_f.config.hidden_size
        self.projector_t = nn.Sequential(nn.Linear(dim_t, 256), nn.BatchNorm1d(256),
                                          nn.ReLU(), nn.Linear(256, 128))
        self.projector_f = nn.Sequential(nn.Linear(dim_f, 256), nn.BatchNorm1d(256),
                                          nn.ReLU(), nn.Linear(256, 128))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder_t.eval()
            self.encoder_f.eval()
        return self

    def _extract(self, encoder, images):
        if self.freeze_backbone:
            with torch.no_grad():
                out = encoder(pixel_values=images)
        else:
            out = encoder(pixel_values=images)
        # HF DINOv3 exposes pooler_output (CLS-based representation).
        features = getattr(out, "pooler_output", None)
        if features is None:
            features = out.last_hidden_state[:, 0]
        return features

    def forward(self, x_in_t, x_in_f):
        # Frequency input is NOW a waveform for STFT, not the old FFT magnitude.
        # See the changed dataloader.py and README_DINOV3.md.
        t_image = self.time_image(x_in_t)
        f_image = self.freq_image(x_in_f)
        h_time = self._extract(self.encoder_t, t_image)
        h_freq = self._extract(self.encoder_f, f_image)
        z_time = self.projector_t(h_time)
        z_freq = self.projector_f(h_freq)
        return h_time, z_time, h_freq, z_freq


class target_classifier(nn.Module):
    """Original TF-C downstream head (128 + 128 -> 64 -> C)."""
    def __init__(self, configs):
        super().__init__()
        self.logits = nn.Linear(256, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        return self.logits_simple(torch.sigmoid(self.logits(emb.reshape(emb.shape[0], -1))))
