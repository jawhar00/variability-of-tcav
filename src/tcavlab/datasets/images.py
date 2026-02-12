from __future__ import annotations
from typing import List
import os, glob, torch
from PIL import Image, ImageFile

# Optional but helpful: allows PIL to load many truncated JPEGs instead of crashing.
ImageFile.LOAD_TRUNCATED_IMAGES = True

def list_image_paths(folder: str, exts=(".jpg",".jpeg",".png",".bmp",".webp")) -> List[str]:
    files=[]
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def load_images_as_tensor(folder: str, preprocess, device: torch.device) -> torch.Tensor:
    paths = list_image_paths(folder)
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")

    imgs = []
    bad = []

    for p in paths:
        try:
            img = Image.open(p)
            img = img.convert("RGB")
            imgs.append(preprocess(img))
        except Exception as e:
            bad.append((p, repr(e)))

    if not imgs:
        raise FileNotFoundError(
            f"All images failed to load in {folder}. "
            f"First error: {bad[0] if bad else 'unknown'}"
        )

    if bad:
        print(f"[load_images_as_tensor] Skipped {len(bad)}/{len(paths)} broken images in {folder}")
        for p, err in bad[:5]:
            print("  -", p, err)

    batch = torch.stack(imgs)
    return batch.to(device)
