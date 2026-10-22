# FairFace download, verification and index (PILOT3.md, Part A).
#
# Source: Hugging Face dataset HuggingFaceM4/FairFace at a pinned revision,
# padding 1.25 parquet shards (train and validation). The file list and the
# LFS sha256 of every shard are read from the Hub API at that revision and
# each downloaded shard is checked against its sha256. Images are decoded
# directly from the parquet bytes; nothing is extracted to disk.
#
# The public release contains the train and validation splits only
# (86,744 + 10,954 = 97,698 images); the remaining images of the 108,501
# described in the paper were not released. Counts are read from the data.
#
# Run:  python -m faceshape_pilot.stage3.fairface

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass

import numpy as np

from faceshape_pilot import config
from faceshape_pilot.common import write_json

API = f"https://huggingface.co/api/datasets/{config.FAIRFACE_HF_REPO}/tree/{config.FAIRFACE_HF_REVISION}"
RESOLVE = f"https://huggingface.co/datasets/{config.FAIRFACE_HF_REPO}/resolve/{config.FAIRFACE_HF_REVISION}"
OUT_JSON = config.STAGE3_DIR / "fairface_download.json"


def shard_list() -> list:
    with urllib.request.urlopen(f"{API}/{config.FAIRFACE_PADDING}") as r:
        entries = json.load(r)
    shards = [{"path": e["path"], "size": e["size"], "sha256": e["lfs"]["oid"]}
              for e in entries if e["path"].endswith(".parquet")]
    return sorted(shards, key=lambda s: s["path"])


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local_path(shard: dict):
    return config.FAIRFACE_DIR / shard["path"]


def download() -> list:
    shards = shard_list()
    for s in shards:
        dst = local_path(s)
        if dst.exists() and dst.stat().st_size == s["size"] and _sha256(dst) == s["sha256"]:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        print(f"[fairface] downloading {s['path']} ({s['size'] / 1e6:.0f} MB)", flush=True)
        with urllib.request.urlopen(f"{RESOLVE}/{s['path']}") as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        got = _sha256(tmp)
        if got != s["sha256"]:
            tmp.unlink()
            raise RuntimeError(f"sha256 mismatch for {s['path']}: {got} != {s['sha256']}")
        tmp.replace(dst)
    return shards


@dataclass
class FairFaceIndex:
    # One row per image, in shard order then row order.
    split: np.ndarray        # "train" / "validation"
    shard: np.ndarray        # shard file name
    row: np.ndarray          # row within shard
    race: np.ndarray         # index into config.FAIRFACE_RACES
    gender: np.ndarray       # index into config.FAIRFACE_GENDERS
    age: np.ndarray          # index into config.FAIRFACE_AGES

    def __len__(self):
        return len(self.split)


def _split_of(path: str) -> str:
    return "validation" if "/validation-" in path else "train"


def load_index(shards=None) -> FairFaceIndex:
    import pyarrow.parquet as pq

    shards = shards or shard_list()
    cols = {k: [] for k in ("split", "shard", "row", "race", "gender", "age")}
    for s in shards:
        t = pq.read_table(local_path(s), columns=["race", "gender", "age"])
        n = t.num_rows
        cols["split"] += [_split_of(s["path"])] * n
        cols["shard"] += [local_path(s).name] * n
        cols["row"].append(np.arange(n))
        for k in ("race", "gender", "age"):
            cols[k].append(t.column(k).to_numpy())
    return FairFaceIndex(
        split=np.array(cols["split"]), shard=np.array(cols["shard"]), row=np.concatenate(cols["row"]),
        race=np.concatenate(cols["race"]).astype(np.int64), gender=np.concatenate(cols["gender"]).astype(np.int64),
        age=np.concatenate(cols["age"]).astype(np.int64),
    )


def iter_images(shards=None, batch_rows: int = 512):
    # Yields (global_index, rgb uint8 array or None if undecodable).
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    shards = shards or shard_list()
    g = 0
    for s in shards:
        pf = pq.ParquetFile(local_path(s))
        for batch in pf.iter_batches(batch_size=batch_rows, columns=["image"]):
            for item in batch.column(0).to_pylist():
                try:
                    with Image.open(io.BytesIO(item["bytes"])) as im:
                        rgb = np.asarray(im.convert("RGB"))
                except Exception:
                    rgb = None
                yield g, rgb
                g += 1


def main():
    shards = download()
    idx = load_index(shards)
    write_json(OUT_JSON, {
        "dataset": "FairFace (Karkkainen and Joo, WACV 2021)",
        "source": f"https://huggingface.co/datasets/{config.FAIRFACE_HF_REPO}",
        "revision": config.FAIRFACE_HF_REVISION,
        "padding": config.FAIRFACE_PADDING,
        "license": "CC BY 4.0 (dataset card)",
        "shards": [{**s, "verified_sha256": True} for s in shards],
        "total_bytes": sum(s["size"] for s in shards),
        "n_images": len(idx),
        "n_by_split": {sp: int((idx.split == sp).sum()) for sp in ("train", "validation")},
        "note": "The public release contains train and validation only; the 108,501 in the paper include "
                "images that were not released.",
        "label_definitions": {"race": config.FAIRFACE_RACES, "gender": config.FAIRFACE_GENDERS,
                              "age": config.FAIRFACE_AGES},
        "label_provenance": "annotator-perceived labels (Amazon Mechanical Turk), not self-identification",
    })
    print(f"[fairface] {len(idx)} images verified -> {OUT_JSON}")


if __name__ == "__main__":
    main()
