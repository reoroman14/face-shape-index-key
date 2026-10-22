# Step 1 (PILOT2.md): a learned face shape router.
#
# Fine-tunes two ImageNet-pretrained backbones (EfficientNet-B0, ResNet-50)
# on the Kaggle Face Shape Dataset training split. 10 percent of each
# training class is held out (fixed seed) as validation; the epoch and the
# backbone are both selected on validation accuracy. The published test
# split is evaluated once, with the selected model only.
#
# Images in which SCRFD finds no face are recorded as detection failures.
# Test accuracy is reported on detected images and, conservatively, with
# every detection failure counted as wrong.
#
# Stages: (1) detection and crops, cached; (2) each backbone trained in its
# own subprocess so no other model or ONNX session shares the 8 GB GPU;
# (3) selection and the single test evaluation.
#
# Run:  python -m faceshape_pilot.stage2.step1_train_cnn

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from faceshape_pilot import config
from faceshape_pilot.common import environment_info, read_json, write_json
from faceshape_pilot.stage2 import cnn, faces, kaggle

OUT_JSON = config.STAGE2_DIR / "cnn_training.json"
OUT_TEST_PROBS = config.STAGE2_DIR / "cnn_kaggle_test_probs.csv"
DET_CACHE = config.STAGE2_CACHE / "kaggle_faces.npz"
CROPS_CACHE = config.STAGE2_CACHE / "kaggle_crops.npy"
PREP_META = config.STAGE2_CACHE / "kaggle_prep.json"
VAL_FRACTION = 0.10
STREAM_VAL_SPLIT = 6


def model_path(name: str) -> Path:
    return config.STAGE2_CACHE / f"cnn_{name}.pt"


def detect_kaggle(index, detector):
    if DET_CACHE.exists():
        c = np.load(DET_CACHE)
        if list(c["paths"]) == [Path(p).relative_to(config.KAGGLE_DIR).as_posix() for p in index.paths]:
            return c["bbox"], c["score"], c["ok"]
    n = len(index.paths)
    bbox, score, ok = np.full((n, 4), np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool)
    for i, p in enumerate(index.paths):
        b, s = detector.detect(faces.limit_side(kaggle.read_rgb(p)))
        if b is not None:
            bbox[i], score[i], ok[i] = b, s, True
    config.STAGE2_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DET_CACHE, paths=np.array([Path(p).relative_to(config.KAGGLE_DIR).as_posix() for p in index.paths]),
                        bbox=bbox, score=score, ok=ok)
    return bbox, score, ok


def make_crops(index, bbox, ok):
    crops = np.zeros((len(index.paths), faces.CROP_SIZE, faces.CROP_SIZE, 3), dtype=np.uint8)
    for i in np.flatnonzero(ok):
        rgb = faces.limit_side(kaggle.read_rgb(index.paths[i]))
        crops[i] = faces.extract_crop(rgb, faces.crop_square(bbox[i]))
    return crops


def val_split(index, ok):
    rng = np.random.default_rng([config.SEED, STREAM_VAL_SPLIT])
    is_val = np.zeros(len(index.paths), dtype=bool)
    for c in range(len(config.KAGGLE_CLASSES)):
        cand = np.flatnonzero((index.split == "train") & (index.label == c) & ok)
        k = int(round(VAL_FRACTION * len(cand)))
        is_val[rng.choice(cand, size=k, replace=False)] = True
    is_train = (index.split == "train") & ok & ~is_val
    return is_train, is_val


def history_path(name: str) -> Path:
    return config.STAGE2_CACHE / f"cnn_{name}_history.json"


def prepare():
    # Stage 1: SCRFD detection and crops; the ONNX session ends with the process.
    index = kaggle.load_index()
    detector = faces.FaceDetector()
    bbox, score, ok = detect_kaggle(index, detector)
    np.save(CROPS_CACHE, make_crops(index, bbox, ok))
    write_json(PREP_META, {"detector_providers_in_use": detector.providers_in_use()})
    print(f"[step1] prepared crops for {int(ok.sum())} of {len(ok)} images", flush=True)


def _arrays():
    index = kaggle.load_index()
    c = np.load(DET_CACHE)
    ok = c["ok"]
    is_train, is_val = val_split(index, ok)
    return index, ok, is_train, is_val, np.load(CROPS_CACHE, mmap_mode="r")


def train_one(name: str):
    # Stage 2: one backbone, alone on the GPU.
    import torch

    index, ok, is_train, is_val, crops = _arrays()
    y = index.label
    state, history, best_epoch = cnn.train(name, np.asarray(crops[is_train]), y[is_train],
                                           np.asarray(crops[is_val]), y[is_val], len(config.KAGGLE_CLASSES),
                                           config.SEED)
    torch.save(state, model_path(name))
    write_json(history_path(name), {"history": history, "best_epoch": best_epoch})
    print(f"[step1] {name}: best val_acc {max(h['val_acc'] for h in history):.4f} at epoch {best_epoch}", flush=True)


def _run(args):
    subprocess.run([sys.executable, "-m", "faceshape_pilot.stage2.step1_train_cnn", *args], check=True)


def main():
    _run(["--prepare"])
    for name in cnn.BACKBONES:
        _run(["--train", name])
    return evaluate()


def evaluate():
    # Stage 3: selection on validation accuracy, single test evaluation.
    index, ok, is_train, is_val, crops = _arrays()
    crops = np.asarray(crops)
    providers = read_json(PREP_META)["detector_providers_in_use"]
    is_test = index.split == "test"
    y = index.label
    n_cls = len(config.KAGGLE_CLASSES)

    runs = {}
    for name in cnn.BACKBONES:
        h = read_json(history_path(name))
        runs[name] = {
            "pretrained_weights": cnn.pretrained_weights_id(name),
            "history": h["history"],
            "best_epoch": h["best_epoch"],
            "best_val_acc": max(e["val_acc"] for e in h["history"]),
        }

    selected = max(cnn.BACKBONES, key=lambda n: (runs[n]["best_val_acc"], -cnn.BACKBONES.index(n)))
    model = cnn.load_trained(selected, n_cls, model_path(selected))

    # Single evaluation on the published test split.
    test_idx = np.flatnonzero(is_test)
    det_test = test_idx[ok[test_idx]]
    probs = np.full((len(index.paths), n_cls), np.nan)
    probs[det_test] = cnn.predict_proba(model, crops[det_test])
    pred = np.full(len(index.paths), -1)
    pred[det_test] = probs[det_test].argmax(1)

    integ = read_json(kaggle.OUT_JSON)
    dup_test = {str(config.KAGGLE_DIR / p) for p in integ["test_images_with_train_near_duplicate"]}
    dup_mask = np.array([Path(p).as_posix() in {Path(d).as_posix() for d in dup_test} for p in index.paths])
    no_dup = det_test[~dup_mask[det_test]]

    from sklearn.metrics import confusion_matrix

    acc_detected = float((pred[det_test] == y[det_test]).mean())
    acc_all = float((pred[test_idx] == y[test_idx]).mean())
    per_class = {c: float((pred[det_test][y[det_test] == k] == k).mean()) for k, c in enumerate(config.KAGGLE_CLASSES)}

    with open(OUT_TEST_PROBS, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "true_label", "face_detected", "pred_label"] + [f"p_{c}" for c in config.KAGGLE_CLASSES])
        for i in test_idx:
            wr.writerow([Path(index.paths[i]).relative_to(config.KAGGLE_DIR).as_posix(), config.KAGGLE_CLASSES[y[i]],
                         bool(ok[i]), config.KAGGLE_CLASSES[pred[i]] if pred[i] >= 0 else ""]
                        + (["" for _ in range(n_cls)] if not ok[i] else [f"{v:.6f}" for v in probs[i]]))

    result = {
        "seed": config.SEED,
        "dataset": {
            "name": "Kaggle Face Shape Dataset (lama2019faceshapedataset)",
            "kaggle_id": config.KAGGLE_DATASET,
            "license": "CC0: Public Domain",
            "integrity_file": kaggle.OUT_JSON.name,
            "n_decoded": len(index.paths),
            "decode_failed": list(index.decode_failed),
        },
        "preprocessing": {
            "detector": "SCRFD (buffalo_l det_10g), det_size 320, face nearest image centre",
            "detector_providers_in_use": providers,
            "kaggle_max_side_px": faces.KAGGLE_MAX_SIDE,
            "crop_scale": faces.CROP_SCALE,
            "crop_up_shift": faces.CROP_UP_SHIFT,
            "crop_size": faces.CROP_SIZE,
            "border": "replicate",
        },
        "detection": {
            split: {"n": int((index.split == split).sum()), "face_found": int(((index.split == split) & ok).sum()),
                    "no_face": int(((index.split == split) & ~ok).sum())}
            for split in ("train", "test")
        },
        "split": {
            "train_used": int(is_train.sum()),
            "validation_held_out_from_train": int(is_val.sum()),
            "validation_fraction_per_class": VAL_FRACTION,
            "validation_seed_stream": [config.SEED, STREAM_VAL_SPLIT],
            "test": int(is_test.sum()),
        },
        "training": {
            "epochs": cnn.EPOCHS, "batch_size": cnn.BATCH_SIZE, "optimizer": "AdamW", "lr": cnn.LR,
            "weight_decay": cnn.WEIGHT_DECAY, "schedule": "cosine annealing over epochs",
            "loss": "cross-entropy", "augmentation": cnn.AUG,
            "determinism": "torch.use_deterministic_algorithms(True, warn_only=True), cudnn.deterministic, "
                           "CUBLAS_WORKSPACE_CONFIG=:4096:8, fixed seed",
            "selection": "epoch and backbone by validation accuracy; test evaluated once, selected model only",
        },
        "runs": runs,
        "selected_backbone": selected,
        "test": {
            "accuracy_on_detected": acc_detected,
            "n_detected": int(len(det_test)),
            "accuracy_counting_detection_failures_as_wrong": acc_all,
            "n_test": int(len(test_idx)),
            "accuracy_excluding_test_images_with_train_duplicate": float((pred[no_dup] == y[no_dup]).mean()),
            "n_excluding_duplicates": int(len(no_dup)),
            "per_class_accuracy_on_detected": per_class,
            "confusion_rows_true_cols_pred": confusion_matrix(y[det_test], pred[det_test], labels=list(range(n_cls))),
            "classes": config.KAGGLE_CLASSES,
            "probabilities_file": OUT_TEST_PROBS.name,
        },
        "published_reference": "up to 96.32 percent on this corpus (grd2024transfer)",
        "model_files": {n: model_path(n).relative_to(config.REPO_ROOT).as_posix() for n in cnn.BACKBONES},
        "environment": environment_info(),
    }
    write_json(OUT_JSON, result)
    print(f"[step1] selected {selected}; test accuracy {acc_detected:.4f} on {len(det_test)} detected, "
          f"{acc_all:.4f} counting failures as wrong  -> {OUT_JSON}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="detection and crops only")
    ap.add_argument("--train", choices=cnn.BACKBONES, help="train one backbone only")
    ap.add_argument("--evaluate", action="store_true", help="selection and test evaluation only")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.train:
        train_one(a.train)
    elif a.evaluate:
        evaluate()
    else:
        main()
