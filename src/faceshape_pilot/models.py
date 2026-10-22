# Model download and hashing. Every model file's SHA-256 is recorded in the
# run manifest so a result can be tied to the exact weights that produced it.

import hashlib
import urllib.request
from pathlib import Path

from faceshape_pilot import config


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_mediapipe_model() -> Path:
    path = config.MEDIAPIPE_MODEL_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(config.MEDIAPIPE_MODEL_URL, tmp)
        tmp.replace(path)
    return path


def model_hashes() -> dict:
    # SHA-256 of every model file present under models/.
    out = {}
    if config.MODELS_DIR.exists():
        for p in sorted(config.MODELS_DIR.rglob("*")):
            if p.is_file() and p.suffix in {".task", ".onnx"}:
                out[str(p.relative_to(config.REPO_ROOT).as_posix())] = sha256_file(p)
    return out
