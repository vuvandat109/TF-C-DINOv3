
import torch
from pathlib import Path
from transformers import AutoModel
from dinov3_adapter import SignalImageAdapter

# 1. Load mot mau EEG
data_path = Path(__file__).resolve().parents[2] / "datasets" / "SleepEEG" / "train.pt"

data = torch.load(
    data_path,
    map_location="cpu",
    weights_only=True
)

x = data["samples"][:1].float()

print("EEG input:", x.shape)

# 2. Chuyen EEG thanh anh
time_adapter = SignalImageAdapter("time")
freq_adapter = SignalImageAdapter("frequency")

time_image = time_adapter(x)
freq_image = freq_adapter(x)

print("Time image:", time_image.shape)
print("Frequency image:", freq_image.shape)

# 3. Load backbone DINOv3 pretrained
model = AutoModel.from_pretrained(
    "facebook/dinov3-vits16-pretrain-lvd1689m"
)
model.eval()

# 4. Trich xuat dac trung tu hai anh
with torch.inference_mode():
    out_time = model(pixel_values=time_image)
    out_freq = model(pixel_values=freq_image)

    h_time = out_time.last_hidden_state[:, 0, :]
    h_freq = out_freq.last_hidden_state[:, 0, :]

print("Time features:", h_time.shape)
print("Frequency features:", h_freq.shape)
print("Time finite:", torch.isfinite(h_time).all().item())
print("Frequency finite:", torch.isfinite(h_freq).all().item())

print("SUCCESS: Real DINOv3 forward passed!")
