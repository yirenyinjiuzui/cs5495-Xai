"""Download the agreed, unmodified model and dataset; verify their contents."""

import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
SOURCES = {
    "resnet18-f37072fd.pth": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "imagenette2-320.tgz": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    "imagenette-project-LICENSE.txt": "https://raw.githubusercontent.com/fastai/imagenette/master/LICENSE",
    "torchvision-project-LICENSE.txt": "https://raw.githubusercontent.com/pytorch/vision/main/LICENSE",
}


def main():
    OUT.mkdir(exist_ok=True)
    records = []
    for name, url in SOURCES.items():
        path = OUT / name
        temporary = path.with_name(name + ".part")
        print(f"Downloading {name}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(path)
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if name.endswith(".pth"):
            if not digest.startswith("f37072fd"):
                raise ValueError("Checkpoint SHA-256 does not match Torchvision's published hash prefix")
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Checkpoint ZIP failed CRC verification")
        record = {"file": name, "source": url, "bytes": path.stat().st_size, "sha256": digest}
        if name.endswith(".tgz"):
            counts = {"train": 0, "val": 0}
            classes = {"train": set(), "val": set()}
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    parts = Path(member.name).parts
                    if member.isfile() and len(parts) == 4 and parts[1] in counts and parts[-1].lower().endswith((".jpg", ".jpeg")):
                        counts[parts[1]] += 1
                        classes[parts[1]].add(parts[2])
            if not all(counts.values()) or any(len(value) != 10 for value in classes.values()) or classes["train"] != classes["val"]:
                raise ValueError("Expected train/val images for the same ten Imagenette classes")
            record.update(images=counts, synsets=sorted(classes["val"]))
        records.append(record)
        print(f"Verified {name}: {record['bytes']} bytes, SHA-256 {digest}", flush=True)
    shutil.copyfile(ROOT / "THIRD_PARTY_NOTICES.md", OUT / "THIRD_PARTY_NOTICES.md")
    records.append({"file": "THIRD_PARTY_NOTICES.md", "sha256": hashlib.sha256((OUT / "THIRD_PARTY_NOTICES.md").read_bytes()).hexdigest()})
    (OUT / "asset_sources.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    checksums = [f"{item['sha256']}  {item['file']}" for item in records]
    checksums.append(f"{hashlib.sha256((OUT / 'asset_sources.json').read_bytes()).hexdigest()}  asset_sources.json")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
