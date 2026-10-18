
import torch
from pathlib import Path
from types import SimpleNamespace

from model import TFC

torch.set_num_threads(2)

# Cấu hình tối thiểu để kiểm tra mô hình
configs = SimpleNamespace(
    num_classes_target=2
)

# Đọc dữ liệu EEG thật
path = (
    Path(__file__).resolve().parents[2]
    / "datasets" / "SleepEEG" / "train.pt"
)

data = torch.load(
    path,
    map_location="cpu",
    weights_only=True
)

x = data["samples"][:2].float()

print("EEG input:", x.shape)

# Khởi tạo hai DINOv3 độc lập
model = TFC(
    configs,
    backbone_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    freeze_backbone=True,
    image_size=224,
    n_fft=64
)

model.eval()

print("Independent encoders:", model.encoder_t is not model.encoder_f)

with torch.inference_mode():
    h_t, z_t, h_f, z_f = model(x, x)
    combined = torch.cat((z_t, z_f), dim=1)

print("Time features:", h_t.shape)
print("Time projector:", z_t.shape)
print("Frequency features:", h_f.shape)
print("Frequency projector:", z_f.shape)
print("Combined:", combined.shape)
print("Finite:", torch.isfinite(combined).all().item())

print("SUCCESS: Both real DINOv3 encoders passed!")
