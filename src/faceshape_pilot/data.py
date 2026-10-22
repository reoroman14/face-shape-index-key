# LFW loading.
#
# sklearn.datasets.fetch_lfw_people is used to download LFW (funneled) into
# data/. Asking it for full-resolution colour arrays would allocate about
# 7 GB, so after the download the JPEG files are indexed directly from the
# sklearn cache and read one at a time.

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from faceshape_pilot import config


@dataclass(frozen=True)
class LFWIndex:
    # Absolute paths, one per image, sorted by identity then file name.
    paths: tuple
    # Integer identity id per image (index into names).
    identity: np.ndarray
    # Identity names, sorted.
    names: tuple
    # Number of images and identities sklearn reports for the same filter.
    sklearn_n_images: int
    sklearn_n_identities: int


def _download_with_sklearn(data_home: Path):
    # Default resize and slice keep the returned array small (about 100 MB);
    # the call is made only to trigger and verify the download.
    from sklearn.datasets import fetch_lfw_people

    bunch = fetch_lfw_people(
        data_home=str(data_home),
        funneled=True,
        min_faces_per_person=config.MIN_FACES_PER_PERSON,
        download_if_missing=True,
    )
    return int(bunch.images.shape[0]), int(len(bunch.target_names))


def load_lfw_index() -> LFWIndex:
    data_home = config.DATA_DIR
    data_home.mkdir(parents=True, exist_ok=True)
    sk_images, sk_ids = _download_with_sklearn(data_home)

    root = data_home / "lfw_home" / "lfw_funneled"
    if not root.is_dir():
        raise FileNotFoundError(f"Expected LFW folder not found: {root}")

    names, paths, identity = [], [], []
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(person_dir.glob("*.jpg"))
        if len(files) < config.MIN_FACES_PER_PERSON:
            continue
        ident = len(names)
        names.append(person_dir.name)
        for f in files:
            paths.append(str(f))
            identity.append(ident)

    index = LFWIndex(
        paths=tuple(paths),
        identity=np.asarray(identity, dtype=np.int64),
        names=tuple(names),
        sklearn_n_images=sk_images,
        sklearn_n_identities=sk_ids,
    )
    if (len(index.paths), len(index.names)) != (sk_images, sk_ids):
        raise RuntimeError(
            "Direct folder index disagrees with sklearn: "
            f"{len(index.paths)} images / {len(index.names)} identities vs "
            f"{sk_images} / {sk_ids}"
        )
    return index


def read_rgb(path: str) -> np.ndarray:
    # Returns an HxWx3 uint8 RGB array.
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)
