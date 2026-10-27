"""Download the public TF-C benchmark archives from Figshare, validate layout.

No Hugging Face token is used here. Datasets are kept locally and ignored by Git.
"""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASETS = {
    "SleepEEG": "https://figshare.com/ndownloader/articles/19930178/versions/1",
    "Epilepsy": "https://figshare.com/ndownloader/articles/19930199/versions/2",
}
REQUIRED = {"SleepEEG": ("train.pt",), "Epilepsy": ("train.pt", "test.pt")}


def download_one(name, root):
    folder = root / name
    if all((folder / f).is_file() for f in REQUIRED[name]):
        print(f"READY: {name} already exists")
        return
    root.mkdir(parents=True, exist_ok=True)
    archive = root / (name + ".zip")
    print(f"Downloading {name} from public Figshare...", flush=True)
    request = urllib.request.Request(DATASETS[name], headers={"User-Agent": "TF-C-DINOv3-CI/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"{archive} is not a ZIP: check Figshare public download URL")
    folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        print(f"ZIP entries ({name}): {[i.filename for i in z.infolist()[:12]]}")
        for info in z.infolist():
            if info.is_dir():
                continue
            target_name = Path(info.filename).name
            if target_name not in REQUIRED[name]:
                continue
            target = folder / target_name
            if target.exists():
                raise ValueError(f"Duplicate archive member named {target_name}")
            with z.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    archive.unlink()
    missing = [f for f in REQUIRED[name] if not (folder / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{name} archive did not contain {missing} as .pt files. "
            "Inspect archive structure/format; do not claim a successful training run."
        )
    print(f"READY: {name}: {[str(folder / f) for f in REQUIRED[name]]}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "datasets")
    args = p.parse_args()
    for name in DATASETS:
        download_one(name, args.root)
