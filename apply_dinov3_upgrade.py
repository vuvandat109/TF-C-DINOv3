"""One-time patch for https://github.com/vuvandat109/TF-C-DINOv3.

Run at the repository ROOT, after copying the package's code/TFC files
into the matching repository folder:
    python apply_dinov3_upgrade.py

Conservative: rejects unknown trainer layouts instead of corrupting files.
Creates *.before_dinov3_loss.bak backups before modifying source files.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
TFC = ROOT / "code" / "TFC"
TRAINER = TFC / "trainer.py"
MAIN = TFC / "main.py"
OLD_LOSS = TFC / "loss.py"
README = ROOT / "README_DINOV3.md"
for p in (TRAINER, MAIN, OLD_LOSS, README, TFC / "dinov3_loss.py", TFC / "test_dinov3_loss.py"):
    if not p.exists():
        sys.exit(f"THIEU FILE: {p}. Hay copy code/TFC/ tu goi nang cap vao dung repo!")

PRETRAIN_BLOCK = '''def _dinov3_loss_from_config(config):
    """Create the DINOv3 TF-C objective on the trainable projected embeddings."""
    return DINOv3TFCLoss(
        temperature=getattr(config, "dinov3_loss_temperature", config.Context_Cont.temperature),
        time_weight=getattr(config, "dinov3_loss_time_weight", 0.2),
        frequency_weight=getattr(config, "dinov3_loss_frequency_weight", 0.2),
        cross_weight=getattr(config, "dinov3_loss_cross_weight", 1.0),
    )


def model_pretrain(model, model_optimizer, criterion, train_loader, config, device, training_mode):
    """Pretrain DINOv3 branch projectors with explicit time/frequency loss."""
    model.train()
    objective = _dinov3_loss_from_config(config)
    batch_losses, time_losses, freq_losses, cross_losses = [], [], [], []

    for batch_idx, (data, labels, aug1, data_f, aug1_f) in enumerate(train_loader):
        if data.shape[0] < 2:
            continue  # contrastive loss requires at least two samples
        model_optimizer.zero_grad(set_to_none=True)
        data, data_f = data.float().to(device), data_f.float().to(device)
        aug1, aug1_f = aug1.float().to(device), aug1_f.float().to(device)

        # DINOv3 backbone features h_* may be frozen. Compute ALL loss terms
        # on z_* so both trainable projectors receive gradients.
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)
        loss, terms = objective(z_t, z_t_aug, z_f, z_f_aug)
        loss.backward()
        model_optimizer.step()

        batch_losses.append(loss.detach().item())
        time_losses.append(terms["time"].item())
        freq_losses.append(terms["frequency"].item())
        cross_losses.append(terms["cross_time_frequency"].item())

    if not batch_losses:
        raise ValueError("Pretraining requires a batch with at least two samples")
    print("DINOv3-TF-C pretrain | total={:.4f} | L_time={:.4f} | L_freq={:.4f} | L_cross={:.4f}".format(
        np.mean(batch_losses), np.mean(time_losses), np.mean(freq_losses), np.mean(cross_losses)
    ))
    return torch.tensor(batch_losses).mean()


'''


def replace_exact(text, old, new, name):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{name}: expected exactly 1 match, found {count}. NOT MODIFIED.")
    return text.replace(old, new, 1)


def patch_trainer(content):
    if "def _dinov3_loss_from_config" in content:
        print("[SKIP] trainer.py da cap nhat")
        return content
    start = content.find("def model_pretrain(")
    end = content.find("def model_finetune(", start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not locate model_pretrain/model_finetune in trainer.py")
    content = content[:start] + PRETRAIN_BLOCK + content[end:]
    # Limit following replacements to fine-tuning, not test function.
    start = content.index("def model_finetune(")
    end = content.index("def model_test(", start)
    segment = content[start:end]
    segment = replace_exact(segment, "    criterion = nn.CrossEntropyLoss()\n",
                            "    criterion = nn.CrossEntropyLoss()\n    objective = _dinov3_loss_from_config(config)\n",
                            "insert fine-tune objective")
    a = segment.find("        nt_xent_criterion = NTXentLoss_poly(")
    b = segment.find('        """Add supervised classifier', a)
    if a < 0 or b < 0:
        raise RuntimeError("Could not identify old NT-Xent fine-tuning block")
    new = '''        # New DINOv3-specific contrastive objective on projected z_* features.
        ssl_loss, ssl_parts = objective(z_t, z_t_aug, z_f, z_f_aug)

'''
    segment = segment[:a] + new + segment[b:]
    segment = replace_exact(segment,
                            "        lam = 0.1\n        loss = loss_p + l_TF + lam*(loss_t + loss_f)\n",
                            '        loss = loss_p + getattr(config, "dinov3_finetune_ssl_weight", 0.1) * ssl_loss\n',
                            "replace fine-tune final loss")
    return content[:start] + segment + content[end:]


def patch_main(content):
    if "--dinov3_loss_cross_weight" in content:
        print("[SKIP] main.py da cap nhat")
        return content
    marker = "parser.add_argument('--subset', action='store_true', help='Use small subset for debugging')"
    extra = '''parser.add_argument('--dinov3_loss_time_weight', default=0.2, type=float,
                    help='Intra-time NT-Xent weight on DINOv3 projected embeddings')
parser.add_argument('--dinov3_loss_frequency_weight', default=0.2, type=float,
                    help='Intra-frequency NT-Xent weight on DINOv3 projected embeddings')
parser.add_argument('--dinov3_loss_cross_weight', default=1.0, type=float,
                    help='Symmetric time-frequency NT-Xent weight')
parser.add_argument('--dinov3_loss_temperature', default=0.2, type=float,
                    help='InfoNCE temperature')
parser.add_argument('--dinov3_finetune_ssl_weight', default=0.1, type=float,
                    help='Weight of self-supervised objective during fine-tuning')'''
    content = replace_exact(content, marker, marker + "\n" + extra, "add CLI options")
    config_marker = "configs = Configs()"
    bind = '''# Expose DINOv3-specific objective coefficients to the training loop.
configs.dinov3_loss_time_weight = args.dinov3_loss_time_weight
configs.dinov3_loss_frequency_weight = args.dinov3_loss_frequency_weight
configs.dinov3_loss_cross_weight = args.dinov3_loss_cross_weight
configs.dinov3_loss_temperature = args.dinov3_loss_temperature
configs.dinov3_finetune_ssl_weight = args.dinov3_finetune_ssl_weight'''
    return replace_exact(content, config_marker, config_marker + "\n" + bind, "add config options")


def patch_loss(content):
    if "from dinov3_loss import DINOv3TFCLoss" in content:
        print("[SKIP] loss.py da cap nhat")
        return content
    return content + ('\n' if not content.endswith('\n') else '') + '''
# Updated DINOv3-TF-C training objective. The old NT-Xent implementations
# above remain for historical comparison, but trainer.py now uses this class.
from dinov3_loss import DINOv3TFCLoss
'''


def patch_readme(content):
    if "## Cập nhật hàm loss cho hai DINOv3" in content:
        print("[SKIP] README_DINOV3.md da cap nhat")
        return content
    block = '''# THAY ĐỔI DINOv3 + LOSS — HƯỚNG DẪN KIỂM TRA CÂU 1

**Đường dẫn code để thầy kiểm tra trực tiếp:**

- [Hai DINOv3 pretrained độc lập: `model.py`](code/TFC/model.py)
- [Chuyển chuỗi tín hiệu thành ảnh: `dinov3_adapter.py`](code/TFC/dinov3_adapter.py)
- [Loss mới: `dinov3_loss.py`](code/TFC/dinov3_loss.py)
- [Hàm `loss.py` trỏ tới loss mới](code/TFC/loss.py)
- [Pre-training/Fine-tuning gọi loss mới: `trainer.py`](code/TFC/trainer.py)
- [Các trọng số cấu hình loss: `main.py`](code/TFC/main.py)
- [Bài kiểm thử loss/gradient: `test_dinov3_loss.py`](code/TFC/test_dinov3_loss.py)

## Cập nhật hàm loss cho hai DINOv3

Mã nguồn gốc có `NTXentLoss_poly` tính `L_t`, `L_f` trên đặc trưng backbone
`h_t`, `h_f`, trong khi hai backbone DINOv3 có thể bị đóng băng. Phiên bản
này chuyển loss sang **đầu ra projector** `z_t`, `z_f` để các thành phần loss
đều có gradient đối với tham số được huấn luyện. Hàm loss mới là
`DINOv3TFCLoss` được viết tại `code/TFC/dinov3_loss.py` và dùng thực tế
trong `model_pretrain()` và `model_finetune()`.

Ký hiệu `t`, `f` cho miền thời gian và tần số; dấu `'` chỉ mẫu tăng cường.
`NTXent(a,b)` là InfoNCE đối xứng có positive cùng chỉ số mẫu:

```
L_time  = NTXent(z_t, z_t')
L_freq  = NTXent(z_f, z_f')
L_cross = 1/4 * ( NTXent(z_t, z_f) + NTXent(z_t, z_f')
                + NTXent(z_t', z_f) + NTXent(z_t', z_f') )
L_pretrain = 0.2*L_time + 0.2*L_freq + 1.0*L_cross
L_finetune = CrossEntropy(classifier([z_t;z_f]), y) + 0.1*L_pretrain
```

Các hệ số và nhiệt độ có thể chỉnh bằng các tham số `--dinov3_loss_*`
trong `code/TFC/main.py`. Đây là **biến thể loss cho DINOv3-TF-C**,
không được trình bày như nguyên bản của bài báo TF-C. Mặc định backbone
DINOv3 được đóng băng; projector và classifier được huấn luyện.

**Kiểm thử trên CPU, không cần tải DINOv3:**

```bash
cd code/TFC
python test_dinov3_loss.py
python smoke_test_dinov3.py
```

Bài kiểm thử xác minh loss hữu hạn, positive alignment, và gradient của
bốn vector projector. Kiểm thử đơn vị không thay thế việc huấn luyện thực tế.
Các kết quả accuracy/epoch được báo cáo trước đây **không phải** kết quả
đo lại sau khi cập nhật loss; muốn báo cáo số mới phải huấn luyện lại.

---

'''
    return block + content


mods = {
    TRAINER: patch_trainer(TRAINER.read_text(encoding="utf-8")),
    MAIN: patch_main(MAIN.read_text(encoding="utf-8")),
    OLD_LOSS: patch_loss(OLD_LOSS.read_text(encoding="utf-8")),
    README: patch_readme(README.read_text(encoding="utf-8")),
}

# Parse new source with Python compiler BEFORE touching the checkout.
for file, data in mods.items():
    if file.suffix == ".py":
        compile(data, str(file), "exec")

for path, patched in mods.items():
    original = path.read_text(encoding="utf-8")
    if patched == original:
        continue
    backup = path.with_suffix(path.suffix + ".before_dinov3_loss.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")
    print("[UPDATED]", path.relative_to(ROOT))
print("HOAN THANH. Kiem thu: cd code/TFC && python test_dinov3_loss.py")
