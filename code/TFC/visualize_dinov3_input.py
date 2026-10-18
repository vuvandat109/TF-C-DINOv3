
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dinov3_adapter import SignalImageAdapter

# 1. Doc mot mau EEG
data_path = Path(__file__).resolve().parents[2] / "datasets" / "SleepEEG" / "train.pt"
data = torch.load(data_path, map_location="cpu", weights_only=True)
x = data["samples"][:1].float()

# 2. Tao anh cho hai encoder
time_adapter = SignalImageAdapter("time")
freq_adapter = SignalImageAdapter("frequency")

with torch.no_grad():
    time_img = time_adapter(x)
    freq_img = freq_adapter(x)

print("EEG:", x.shape)
print("Time image:", time_img.shape)
print("Frequency image:", freq_img.shape)

# 3. Hoan tac chuan hoa ImageNet de xem anh
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def to_image(tensor):
    img = tensor[0].detach().cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()

# 4. Luu anh
output = Path(__file__).resolve().parent / "visualizations"
output.mkdir(exist_ok=True)

plt.imsave(output / "time_image.png", to_image(time_img))
plt.imsave(output / "frequency_image.png", to_image(freq_img))

print("Saved images to:", output)
