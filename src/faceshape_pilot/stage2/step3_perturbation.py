# Step 3 (PILOT2.md): in-domain perturbation control.
#
# Kaggle TEST images (in-domain for the CNN) are perturbed with variation
# that two photographs of one person would show but that does not change
# bone structure, and the CNN's label is tracked as a function of strength.
#
# Perturbations, applied in this order to the full image (all ranges at
# strength s = 1; at strength s every range is multiplied by s):
#
#   expression (landmark-guided smooth displacement field, MediaPipe points)
#     jaw drop     : everything below the mouth line, within the face width,
#                    moves down by JAW_DROP_MAX x face height (uniform in
#                    [0, s x max]); fades out below the chin
#     mouth corners: points 61 and 291 move by up to MOUTH_CORNER_MAX x face
#                    height (uniform in [-s, s] x max; positive = outward and
#                    up as in a smile, negative = inward and down), Gaussian
#                    support of MOUTH_SIGMA x face width
#   pose (planar homography; the face is a plane 15 cm wide, 1 m from the
#   camera, rotated about its centre)
#     yaw          : uniform in [-s, s] x YAW_MAX_DEG
#     pitch        : uniform in [-s, s] x PITCH_MAX_DEG
#   lighting (on intensities in [0, 1])
#     gradient     : multiply by 1 + g x clip(projection / face width, -1, 1),
#                    g uniform in [-s, s] x GRADIENT_MAX, direction uniform
#                    in [0, 360) degrees (direction is not strength-scaled)
#     brightness   : multiply by 1 + b, b uniform in [-s, s] x BRIGHTNESS_MAX
#     contrast     : (x - mean) x c + mean, log c uniform in [-s, s] x log(CONTRAST_MAX)
#     gamma        : x ** gamma, log gamma uniform in [-s, s] x log(GAMMA_MAX)
#   crop (after re-detecting the face on the perturbed image)
#     centre shift : dx, dy uniform in [-s, s] x CROP_SHIFT_MAX x crop side
#     scale        : 1 + ds, ds uniform in [-s, s] x CROP_SCALE_MAX
#
# The planar pose model omits 3D effects (nose occlusion, change of the
# visible cheek contour), so it is milder than a real head turn.
#
# For each image, strength and copy the random draws come from
# default_rng([SEED, STREAM, image, strength x 100, copy]); every draw is
# written to perturbation_params.csv. Single-factor conditions at s = 1 reuse
# the full-strength draws with the other factors set to zero.
#
# Metrics, per condition, bootstrapped over images:
#   survival       : fraction of perturbed copies whose top-1 equals the
#                    top-1 on the unperturbed image
#   pair agreement : copies are paired (0,1), (2,3), ...; two perturbed
#                    copies play the role of two photographs of one person.
#                    Agreement and Cohen's kappa (both orders counted, as in
#                    stage 1) are directly comparable with the LFW kappa
#   accuracy       : against the published label
# Copies in which SCRFD finds no face are recorded and excluded.
#
# Run:  python -m faceshape_pilot.stage2.step3_perturbation

import csv
from pathlib import Path

import cv2
import numpy as np

from faceshape_pilot import config, router
from faceshape_pilot.common import environment_info, percentile_ci, read_json, setup_matplotlib, write_json, \
    COLOR_KMEANS, COLOR_SHAPE, COLOR_TEXT_2
from faceshape_pilot.models import ensure_mediapipe_model
from faceshape_pilot.stage2 import cnn, faces, kaggle
from faceshape_pilot.stage2.step1_train_cnn import OUT_JSON as STEP1_JSON, model_path, OUT_TEST_PROBS

OUT_JSON = config.STAGE2_DIR / "perturbation.json"
OUT_PARAMS = config.STAGE2_DIR / "perturbation_params.csv"
OUT_PLOT = config.STAGE2_DIR / "perturbation_curve.png"
OUT_EXAMPLES = config.STAGE2_DIR / "perturbation_examples.jpg"

STREAM = 8
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_COPIES = 10

YAW_MAX_DEG = 15.0
PITCH_MAX_DEG = 10.0
JAW_DROP_MAX = 0.04
MOUTH_CORNER_MAX = 0.03
MOUTH_SIGMA = 0.07
GRADIENT_MAX = 0.25
BRIGHTNESS_MAX = 0.20
CONTRAST_MAX = 1.25
GAMMA_MAX = 1.25
CROP_SHIFT_MAX = 0.04
CROP_SCALE_MAX = 0.05
FACE_WIDTH_M = 0.15
CAMERA_DISTANCE_M = 1.0

FACTORS = {
    "pose": ("yaw_deg", "pitch_deg"),
    "expression": ("jaw_drop", "mouth_corner"),
    "lighting": ("gradient", "brightness", "log_contrast", "log_gamma"),
    "crop": ("crop_dx", "crop_dy", "crop_ds"),
}
PARAM_NAMES = ("yaw_deg", "pitch_deg", "jaw_drop", "mouth_corner", "gradient", "gradient_angle_deg",
               "brightness", "log_contrast", "log_gamma", "crop_dx", "crop_dy", "crop_ds")


def draw_params(image: int, s: float, copy: int) -> dict:
    rng = np.random.default_rng([config.SEED, STREAM, image, int(round(s * 100)), copy])
    u = rng.uniform(-1.0, 1.0, size=11)
    angle = rng.uniform(0.0, 360.0)
    return {
        "yaw_deg": s * YAW_MAX_DEG * u[0],
        "pitch_deg": s * PITCH_MAX_DEG * u[1],
        "jaw_drop": s * JAW_DROP_MAX * (u[2] + 1.0) / 2.0,
        "mouth_corner": s * MOUTH_CORNER_MAX * u[3],
        "gradient": s * GRADIENT_MAX * u[4],
        "gradient_angle_deg": angle,
        "brightness": s * BRIGHTNESS_MAX * u[5],
        "log_contrast": s * np.log(CONTRAST_MAX) * u[6],
        "log_gamma": s * np.log(GAMMA_MAX) * u[7],
        "crop_dx": s * CROP_SHIFT_MAX * u[8],
        "crop_dy": s * CROP_SHIFT_MAX * u[9],
        "crop_ds": s * CROP_SCALE_MAX * u[10],
    }


def only_factor(p: dict, factor: str) -> dict:
    keep = set(FACTORS[factor]) | ({"gradient_angle_deg"} if factor == "lighting" else set())
    return {k: (v if k in keep else 0.0) for k, v in p.items()}


def face_geometry(pts: np.ndarray) -> dict:
    p = pts[:, :2]
    return {
        "centre": p.mean(axis=0),
        "height": float(np.linalg.norm(p[10] - p[152])),
        "width": float(np.linalg.norm(p[234] - p[454])),
        "mouth_y": float((p[13, 1] + p[14, 1]) / 2.0),
        "chin_y": float(p[152, 1]),
        "corners": (p[61], p[291]),
    }


def expression_warp(rgb, geo, jaw, corner):
    if jaw == 0.0 and corner == 0.0:
        return rgb
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    H, W = geo["height"], geo["width"]
    cx = geo["centre"][0]
    if jaw != 0.0:
        below_mouth = 1.0 / (1.0 + np.exp(-(yy - geo["mouth_y"]) / (0.03 * H)))
        lateral = np.exp(-((xx - cx) ** 2) / (2 * (0.45 * W) ** 2))
        fade = np.exp(-np.maximum(0.0, yy - geo["chin_y"]) ** 2 / (2 * (0.15 * H) ** 2))
        dy += jaw * H * below_mouth * lateral * fade
    if corner != 0.0:
        for c, side in zip(geo["corners"], (-1.0, 1.0)):
            g = np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * (MOUTH_SIGMA * W) ** 2))
            dx += corner * H * 0.8 * side * g
            dy += corner * H * -0.6 * g
    # Backward map: output pixel x samples the source at x - D(x).
    return cv2.remap(rgb, xx - dx, yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def pose_warp(rgb, geo, yaw_deg, pitch_deg):
    if yaw_deg == 0.0 and pitch_deg == 0.0:
        return rgb
    cx, cy = geo["centre"]
    f = geo["width"] * CAMERA_DISTANCE_M / FACE_WIDTH_M
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    r = ry @ rx
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    plane = np.column_stack([r[:, 0], r[:, 1], [0.0, 0.0, CAMERA_DISTANCE_M]])
    to_plane = np.diag([CAMERA_DISTANCE_M / f, CAMERA_DISTANCE_M / f, 1.0]) @ np.array(
        [[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    hmat = k @ plane @ to_plane
    hmat /= hmat[2, 2]
    h, w = rgb.shape[:2]
    return cv2.warpPerspective(rgb, hmat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def lighting(rgb, geo, p):
    if all(p[k] == 0.0 for k in FACTORS["lighting"]):
        return rgb
    x = rgb.astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    if p["gradient"] != 0.0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = np.radians(p["gradient_angle_deg"])
        proj = ((xx - geo["centre"][0]) * np.cos(t) + (yy - geo["centre"][1]) * np.sin(t)) / geo["width"]
        x = x * (1.0 + p["gradient"] * np.clip(proj, -1.0, 1.0))[..., None]
    x = x * (1.0 + p["brightness"])
    m = x.mean()
    x = (x - m) * np.exp(p["log_contrast"]) + m
    x = np.clip(x, 0.0, 1.0) ** np.exp(p["log_gamma"])
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def perturb(rgb, geo, p):
    out = expression_warp(rgb, geo, p["jaw_drop"], p["mouth_corner"])
    out = pose_warp(out, geo, p["yaw_deg"], p["pitch_deg"])
    return lighting(out, geo, p)


def cnn_input(rgb, detector, p):
    b, _ = detector.detect(rgb)
    if b is None:
        return None
    return faces.extract_crop(rgb, faces.crop_square(b), (p["crop_dx"], p["crop_dy"], p["crop_ds"]))


def conditions():
    out = [(f"strength_{s:.2f}", s, None) for s in STRENGTHS]
    out += [(f"only_{f}_strength_1.00", 1.0, f) for f in FACTORS]
    return out


def summarise(labels, clean, truth, valid, n_cls, rng_seed):
    # labels/valid: (n_img, N_COPIES); clean/truth: (n_img,)
    n_img = len(clean)
    surv_i = np.where(valid, labels == clean[:, None], False).sum(1)
    acc_i = np.where(valid, labels == truth[:, None], False).sum(1)
    nv_i = valid.sum(1)
    a, b = labels[:, 0::2], labels[:, 1::2]
    pv = valid[:, 0::2] & valid[:, 1::2]
    same_i = np.where(pv, a == b, False).sum(1)
    np_i = pv.sum(1)
    conf_i = np.zeros((n_img, n_cls, n_cls))
    for i in range(n_img):
        for x, y in zip(a[i][pv[i]], b[i][pv[i]]):
            conf_i[i, x, y] += 1
            conf_i[i, y, x] += 1

    def stats(wt):
        with np.errstate(invalid="ignore", divide="ignore"):
            surv = (wt @ surv_i) / (wt @ nv_i)
            acc = (wt @ acc_i) / (wt @ nv_i)
            agree = (wt @ same_i) / (wt @ np_i)
            cm = np.tensordot(wt, conf_i, axes=(-1, 0))
            n = cm.sum(axis=(-1, -2))
            po = np.trace(cm, axis1=-2, axis2=-1) / n
            pe = (cm.sum(-1) * cm.sum(-2)).sum(-1) / n**2
            kappa = (po - pe) / (1 - pe)
        return surv, acc, agree, kappa, pe

    point = stats(np.ones(n_img))
    rng = np.random.default_rng(rng_seed)
    w = rng.multinomial(n_img, np.full(n_img, 1.0 / n_img), size=config.N_BOOTSTRAP).astype(np.float64)
    boot = stats(w)
    names = ("survival", "accuracy", "pair_agreement", "pair_kappa", "pair_chance_agreement")
    out = {}
    for name, v, bv in zip(names, point, boot):
        lo, hi = percentile_ci(bv)
        out[name] = v
        out[f"{name}_ci95"] = {"low": lo, "high": hi}
    out["n_copies_valid"] = int(nv_i.sum())
    out["n_copies_no_face"] = int((~valid).sum())
    out["n_pairs_valid"] = int(np_i.sum())
    return out


def main():
    step1 = read_json(STEP1_JSON)
    selected = step1["selected_backbone"]
    n_cls = len(config.KAGGLE_CLASSES)
    index = kaggle.load_index()
    test_idx = np.flatnonzero(index.split == "test")

    detector = faces.FaceDetector()
    providers = detector.providers_in_use()
    landmarker = router.LandmarkDetector(ensure_mediapipe_model())
    model = cnn.load_trained(selected, n_cls, model_path(selected))

    images, geos, excluded = [], [], {"no_face_clean": 0, "no_landmarks_clean": 0}
    for i in test_idx:
        rgb = faces.limit_side(kaggle.read_rgb(index.paths[i]))
        b, _ = detector.detect(rgb)
        if b is None:
            excluded["no_face_clean"] += 1
            continue
        pts = landmarker.detect(rgb)
        if pts is None:
            excluded["no_landmarks_clean"] += 1
            continue
        images.append(i)
        geos.append(face_geometry(pts))
    landmarker.close()
    images = np.array(images)
    truth = index.label[images]

    zero = {k: 0.0 for k in PARAM_NAMES}
    clean_crops = np.stack([cnn_input(faces.limit_side(kaggle.read_rgb(index.paths[i])), detector, zero)
                            for i in images])
    clean = cnn.predict_proba(model, clean_crops).argmax(1)

    step1_pred = {}
    with open(OUT_TEST_PROBS, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step1_pred[row["path"]] = row["pred_label"]
    clean_matches_step1 = all(
        step1_pred[Path(index.paths[i]).relative_to(config.KAGGLE_DIR).as_posix()] == config.KAGGLE_CLASSES[c]
        for i, c in zip(images, clean))

    conds = conditions()
    results, examples = {}, {}
    example_rows = images[:4]
    with open(OUT_PARAMS, "w", newline="", encoding="utf-8") as fparams:
        wr = csv.writer(fparams)
        wr.writerow(["condition", "path", "copy"] + list(PARAM_NAMES) + ["face_found", "pred_label"])
        for cname, s, factor in conds:
            labels = np.full((len(images), N_COPIES), -1)
            valid = np.zeros((len(images), N_COPIES), dtype=bool)
            for r, (i, geo) in enumerate(zip(images, geos)):
                rgb = faces.limit_side(kaggle.read_rgb(index.paths[i]))
                params, crops, slots = [], [], []
                for j in range(N_COPIES):
                    p = draw_params(int(i), s, j)
                    if factor is not None:
                        p = only_factor(p, factor)
                    params.append(p)
                    c = cnn_input(perturb(rgb, geo, p), detector, p)
                    if c is not None:
                        crops.append(c)
                        slots.append(j)
                if crops:
                    pred = cnn.predict_proba(model, np.stack(crops)).argmax(1)
                    labels[r, slots] = pred
                    valid[r, slots] = True
                    if i in example_rows and 0 in slots:
                        examples[(int(i), cname)] = crops[slots.index(0)]
                relp = Path(index.paths[i]).relative_to(config.KAGGLE_DIR).as_posix()
                for j, p in enumerate(params):
                    wr.writerow([cname, relp, j] + [f"{p[k]:.6f}" for k in PARAM_NAMES]
                                + [bool(valid[r, j]), config.KAGGLE_CLASSES[labels[r, j]] if valid[r, j] else ""])
            results[cname] = {"strength": s, "single_factor": factor,
                              **summarise(labels, clean, truth, valid, n_cls, [config.SEED, STREAM, 99])}
            print(f"[step3] {cname}: survival {results[cname]['survival']:.4f} "
                  f"pair_kappa {results[cname]['pair_kappa']:.4f} accuracy {results[cname]['accuracy']:.4f}",
                  flush=True)

    result = {
        "seed": config.SEED,
        "random_stream": [config.SEED, STREAM, "image", "strength x 100", "copy"],
        "router": {"backbone": selected, "model_file": model_path(selected).relative_to(config.REPO_ROOT).as_posix()},
        "detector_providers_in_use": providers,
        "images": {"kaggle_test_total": int(len(test_idx)), "used": int(len(images)), "excluded": excluded},
        "clean_predictions_match_step1": bool(clean_matches_step1),
        "clean_accuracy_on_used_images": float((clean == truth).mean()),
        "parameters_at_strength_1": {
            "yaw_deg_max_abs": YAW_MAX_DEG, "pitch_deg_max_abs": PITCH_MAX_DEG,
            "pose_model": f"planar face {FACE_WIDTH_M} m wide at {CAMERA_DISTANCE_M} m, rotated about its centre",
            "jaw_drop_max_fraction_of_face_height": JAW_DROP_MAX,
            "mouth_corner_max_abs_fraction_of_face_height": MOUTH_CORNER_MAX,
            "mouth_corner_sigma_fraction_of_face_width": MOUTH_SIGMA,
            "lighting_gradient_max_abs": GRADIENT_MAX, "brightness_max_abs": BRIGHTNESS_MAX,
            "contrast_factor_range": [1 / CONTRAST_MAX, CONTRAST_MAX], "gamma_range": [1 / GAMMA_MAX, GAMMA_MAX],
            "crop_shift_max_abs_fraction_of_side": CROP_SHIFT_MAX, "crop_scale_max_abs": CROP_SCALE_MAX,
            "sampling": "each parameter uniform in [-s, s] x max (jaw drop in [0, s x max]; contrast and gamma "
                        "uniform in log space); gradient direction uniform in [0, 360) at every s > 0",
            "order": "expression -> pose -> lighting -> SCRFD re-detection -> crop jitter",
        },
        "strengths": STRENGTHS,
        "copies_per_image_per_condition": N_COPIES,
        "conditions": results,
        "params_file": OUT_PARAMS.name,
        "examples_file": OUT_EXAMPLES.name,
        "notes": [
            "Training augmentation (crop, flip, rotation up to 10 degrees, colour jitter) overlaps with these "
            "perturbations, which makes the CNN more robust to them; any bias favours label stability.",
            "The planar pose model omits 3D effects, so pose perturbations are milder than real head turns.",
        ],
        "environment": environment_info(),
    }
    write_json(OUT_JSON, result)
    save_examples(examples, example_rows, conds)
    plot(result, OUT_PLOT)
    return result


def save_examples(examples, rows, conds):
    names = [c[0] for c in conds]
    tile = faces.CROP_SIZE // 2
    sheet = np.full((len(rows) * tile + 24, len(names) * tile, 3), 255, np.uint8)
    for r, i in enumerate(rows):
        for c, name in enumerate(names):
            im = examples.get((int(i), name))
            if im is not None:
                sheet[24 + r * tile:24 + (r + 1) * tile, c * tile:(c + 1) * tile] = cv2.resize(im, (tile, tile),
                                                                                              interpolation=cv2.INTER_AREA)
    for c, name in enumerate(names):
        label = name.replace("strength_", "s=").replace("only_", "").replace("_s=1.00", " s=1")
        cv2.putText(sheet, label, (c * tile + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_EXAMPLES), sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])


def plot(result, path):
    plt = setup_matplotlib()
    lfw = read_json(config.STAGE2_DIR / "consistency_cnn_lfw.json") if (config.STAGE2_DIR / "consistency_cnn_lfw.json").exists() else None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    cs = [result["conditions"][f"strength_{s:.2f}"] for s in STRENGTHS]
    for key, color, marker, label in (("survival", COLOR_SHAPE, "o", "Label survival (vs unperturbed)"),
                                      ("pair_kappa", COLOR_KMEANS, "s", "Kappa, two perturbed copies")):
        y = np.array([c[key] for c in cs])
        lo = np.array([c[f"{key}_ci95"]["low"] for c in cs])
        hi = np.array([c[f"{key}_ci95"]["high"] for c in cs])
        ax.fill_between(STRENGTHS, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(STRENGTHS, y, "-", marker=marker, color=color, label=label, markeredgecolor="#fcfcfb",
                markeredgewidth=1.5)
    if lfw is not None:
        k = lfw["cohen_kappa_top1"]
        ax.axhline(k, color=COLOR_TEXT_2, linestyle="--", linewidth=1.2)
        ax.annotate(f"CNN kappa on LFW identities: {k:.3f}", (0.0, k), textcoords="offset points", xytext=(4, 5),
                    fontsize=8, color=COLOR_TEXT_2)
    ax.set_xlabel("Perturbation strength s (1 = full ranges, see perturbation.json)")
    ax.set_ylabel("Value")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(STRENGTHS)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title("Kaggle test images (in-domain), CNN router", fontsize=10)

    ax = axes[1]
    names = list(FACTORS) + ["all"]
    vals = [result["conditions"][f"only_{f}_strength_1.00"]["pair_kappa"] for f in FACTORS] + \
           [result["conditions"]["strength_1.00"]["pair_kappa"]]
    x = np.arange(len(names))
    ax.bar(x, vals, width=0.65, color=COLOR_KMEANS, edgecolor="none")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=COLOR_TEXT_2)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Kappa, two perturbed copies")
    ax.set_title("One factor at a time, s = 1", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
