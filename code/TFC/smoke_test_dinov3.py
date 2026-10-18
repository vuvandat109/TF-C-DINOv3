"""Lightweight no-download adapter/model test, with fake HF AutoModel.
Not a test of real DINOv3 weights or end-to-end training.
"""
from types import SimpleNamespace
import sys
import types
import torch
from torch import nn
from dinov3_adapter import SignalImageAdapter


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.proj = nn.Linear(3, 16)

    def forward(self, pixel_values):
        z = pixel_values.mean(dim=(2, 3))
        return SimpleNamespace(pooler_output=self.proj(z))


class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, model_name):
        return FakeDino()


def main():
    # Mock ONLY model downloads; exercise actual TF-C code, inputs, projectors.
    sys.modules['transformers'] = types.SimpleNamespace(AutoModel=FakeAutoModel)
    from model import TFC, target_classifier
    torch.manual_seed(7)
    x = torch.randn(2, 1, 178)
    time_img = SignalImageAdapter('time')(x)
    freq_img = SignalImageAdapter('frequency')(x)
    assert time_img.shape == (2, 3, 224, 224)
    assert freq_img.shape == (2, 3, 224, 224)
    assert torch.isfinite(time_img).all() and torch.isfinite(freq_img).all()
    model = TFC(SimpleNamespace(num_classes_target=2), freeze_backbone=True)
    assert model.encoder_t is not model.encoder_f
    h_t, z_t, h_f, z_f = model(x, x)
    assert h_t.shape == h_f.shape == (2, 16)
    assert z_t.shape == z_f.shape == (2, 128)
    classifier = target_classifier(SimpleNamespace(num_classes_target=2))
    pred = classifier(torch.cat([z_t, z_f], 1))
    assert pred.shape == (2, 2)
    pred.square().mean().backward()
    assert model.projector_t[0].weight.grad is not None
    assert model.projector_f[0].weight.grad is not None
    assert model.encoder_t.proj.weight.grad is None
    assert model.encoder_f.proj.weight.grad is None
    print('PASS: both DINOv3 encoder slots replaced, adapters, output shapes and projector gradients')


if __name__ == '__main__':
    main()
