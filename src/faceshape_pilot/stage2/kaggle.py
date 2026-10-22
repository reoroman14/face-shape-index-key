# Kaggle Face Shape Dataset (Lama 2019, niten19/face-shape-dataset, CC0):
# 5,000 images of female celebrities, five classes, published split of
# 800 training and 200 testing images per class.
#
# Downloaded anonymously from the Kaggle API into data/kaggle_faceshape/,
# extracted with a path-traversal guard, and indexed. Integrity checks
# (decode failures, image modes, exact and near duplicates within and
# across the split) are recorded; the published split is never modified.
#
# Run:  python -m faceshape_pilot.stage2.kaggle

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from faceshape_pilot import config
from faceshape_pilot.common import write_json
from faceshape_pilot.models import sha256_file

DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{config.KAGGLE_DATASET}"
ARCHIVE = config.KAGGLE_DIR / "archive.zip"
EXTRACT_DIR = config.KAGGLE_DIR / "extracted"
ROOT = EXTRACT_DIR / "FaceShape Dataset"
SPLIT_DIRS = {"train": "training_set", "test": "testing_set"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
# Near duplicates: candidate pairs from a 64-bit difference hash (Hamming
# distance), confirmed by normalised cross-correlation of 64x64 greyscale
# thumbnails. The hash alone matches different people photographed with
# similar framing (checked visually), so it is never used on its own.
NEAR_DUP_HAMMING = 10
NEAR_DUP_NCC = 0.97
OUT_JSON = config.STAGE2_DIR / "kaggle_integrity.json"


@dataclass(frozen=True)
class KaggleIndex:
    paths: tuple
    label: np.ndarray
    split: np.ndarray
    decode_failed: tuple


def ensure_downloaded():
    config.KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        tmp = ARCHIVE.with_suffix(".part")
        urllib.request.urlretrieve(DOWNLOAD_URL, tmp)
        tmp.replace(ARCHIVE)
    if not ROOT.is_dir():
        dest = EXTRACT_DIR.resolve()
        with zipfile.ZipFile(ARCHIVE) as z:
            for member in z.infolist():
                target = (dest / member.filename).resolve()
                if not target.is_relative_to(dest):
                    raise RuntimeError(f"Unsafe path in archive: {member.filename}")
            z.extractall(dest)


def _decodes(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except Exception:
        return False


def load_index() -> KaggleIndex:
    ensure_downloaded()
    paths, labels, splits, failed = [], [], [], []
    for split, sub in SPLIT_DIRS.items():
        for c, name in enumerate(config.KAGGLE_CLASSES):
            folder = ROOT / sub / name.capitalize()
            for f in sorted(folder.iterdir(), key=lambda p: p.name):
                if f.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if not _decodes(f):
                    failed.append(f.relative_to(config.KAGGLE_DIR).as_posix())
                    continue
                paths.append(str(f))
                labels.append(c)
                splits.append(split)
    return KaggleIndex(tuple(paths), np.asarray(labels), np.asarray(splits), tuple(failed))


def read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _thumb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float64).ravel()
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-12)


def _dhash(path: str) -> int:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def integrity(index: KaggleIndex) -> dict:
    n = len(index.paths)
    sha = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in index.paths]
    dh = np.array([_dhash(p) for p in index.paths], dtype=np.uint64)
    sizes, modes = [], {}
    for p in index.paths:
        with Image.open(p) as im:
            sizes.append(im.size)
            modes[im.mode] = modes.get(im.mode, 0) + 1
    sizes = np.array(sizes)

    # exact duplicates
    groups = {}
    for i, h in enumerate(sha):
        groups.setdefault(h, []).append(i)
    exact = [g for g in groups.values() if len(g) > 1]

    def split_kind(g):
        s = set(index.split[g])
        return "train-test" if len(s) > 1 else f"within-{s.pop()}"

    def label_kind(g):
        return "same-label" if len(set(index.label[g])) == 1 else "conflicting-labels"

    # near duplicates (pairwise Hamming distance on dHash)
    bits = np.unpackbits(dh.view(np.uint8).reshape(n, 8), axis=1).astype(bool)
    thumbs = {}
    near_pairs, n_candidates = [], 0
    for i in range(n):
        d = (bits[i + 1:] != bits[i]).sum(axis=1)
        for j in np.flatnonzero(d <= NEAR_DUP_HAMMING) + i + 1:
            n_candidates += 1
            for k in (i, int(j)):
                if k not in thumbs:
                    thumbs[k] = _thumb(index.paths[k])
            ncc = float(thumbs[i] @ thumbs[int(j)])
            if ncc >= NEAR_DUP_NCC:
                near_pairs.append((i, int(j), round(ncc, 4)))
    test_idx = np.flatnonzero(index.split == "test")
    test_with_train_near_dup = sorted({
        a if index.split[a] == "test" else b
        for a, b, _ in near_pairs
        if index.split[a] != index.split[b]
    })

    def rel(i):
        return Path(index.paths[i]).relative_to(config.KAGGLE_DIR).as_posix()

    return {
        "archive_sha256": sha256_file(ARCHIVE),
        "n_images_decoded": n,
        "decode_failed": list(index.decode_failed),
        "counts": {
            split: {c: int(((index.split == split) & (index.label == k)).sum())
                    for k, c in enumerate(config.KAGGLE_CLASSES)}
            for split in SPLIT_DIRS
        },
        "image_modes": modes,
        "image_size_px": {
            "width_min": int(sizes[:, 0].min()), "width_median": float(np.median(sizes[:, 0])),
            "width_max": int(sizes[:, 0].max()),
            "height_min": int(sizes[:, 1].min()), "height_median": float(np.median(sizes[:, 1])),
            "height_max": int(sizes[:, 1].max()),
        },
        "exact_duplicates": {
            "n_groups": len(exact),
            "by_split": {k: sum(1 for g in exact if split_kind(g) == k)
                         for k in ("train-test", "within-train", "within-test")},
            "groups_with_conflicting_labels": sum(1 for g in exact if label_kind(g) == "conflicting-labels"),
            "groups": [[rel(i) for i in g] for g in exact],
        },
        "near_duplicates": {
            "method": "dHash Hamming <= threshold as candidate, confirmed by 64x64 greyscale NCC >= ncc_threshold",
            "hamming_threshold": NEAR_DUP_HAMMING,
            "ncc_threshold": NEAR_DUP_NCC,
            "n_candidate_pairs": n_candidates,
            "note": "Confirmed pairs were inspected visually and are true duplicates. NCC on full-frame "
                    "thumbnails misses duplicates that were re-cropped or re-scaled, so counts are a lower bound.",
            "n_pairs": len(near_pairs),
            "n_train_test_pairs": sum(1 for a, b, _ in near_pairs if index.split[a] != index.split[b]),
            "n_test_images_with_a_train_near_duplicate": len(test_with_train_near_dup),
            "n_test_images": int(len(test_idx)),
            "n_pairs_with_conflicting_labels": sum(1 for a, b, _ in near_pairs if index.label[a] != index.label[b]),
            "pairs": [[rel(a), rel(b), d] for a, b, d in near_pairs],
        },
        "test_images_with_train_near_duplicate": [rel(i) for i in test_with_train_near_dup],
    }


def main():
    index = load_index()
    result = integrity(index)
    write_json(OUT_JSON, result)
    e, nd = result["exact_duplicates"], result["near_duplicates"]
    print(f"[kaggle] {result['n_images_decoded']} images, {len(result['decode_failed'])} failed to decode; "
          f"exact dup groups {e['n_groups']} {e['by_split']}; near-dup pairs {nd['n_pairs']} "
          f"(train-test {nd['n_train_test_pairs']}, test images affected "
          f"{nd['n_test_images_with_a_train_near_duplicate']})  -> {OUT_JSON}")
    return index, result


if __name__ == "__main__":
    main()
