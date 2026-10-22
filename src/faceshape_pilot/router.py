# Landmark face shape router.
#
# MediaPipe Face Landmarker (CPU) gives 478 points; the first 468 follow the
# Face Mesh topology used here. Five measurements are taken in 3D (x, y in
# pixels, z on the same scale as x) to reduce the effect of head rotation:
#
#   forehead width  : points 54 - 284   (upper forehead, on the face oval)
#   cheekbone width : points 234 - 454  (widest points of the face oval)
#   jaw width       : points 172 - 397  (gonion region of the face oval)
#   face length     : points 10 - 152   (upper forehead midline to chin)
#   jaw angle       : angle at the gonion (172 / 397) between the ramus
#                     direction (towards 234 / 454) and the chin (152),
#                     mean of left and right, in degrees
#
# Point 10 lies on the upper forehead, not the hairline, so absolute
# length/width thresholds from cosmetic guides do not transfer. The router
# therefore works on four scale-free features standardised with robust
# statistics (median and IQR) over the pilot images:
#
#   L = face length / cheekbone width
#   F = forehead width / cheekbone width
#   J = jaw width / cheekbone width
#   A = jaw angle (larger = more obtuse, softer jaw)
#
# Standardisation uses no identity labels. Each class is defined by one
# explicit geometric rule, written as a prototype point in (L, F, J, A)
# standard-score space. All prototypes constrain all four features with the
# same width, so no class is favoured by having fewer constraints. The class
# score is -0.5 * squared distance to the prototype and the output is the
# softmax over classes: a probability distribution, not a hard label.
#
# These prototypes were fixed before any measurement was run and must not be
# tuned against the pilot outcomes; tuning them to raise identity consistency
# would amount to training the router on identity labels.

import numpy as np

from faceshape_pilot import config

IDX_FOREHEAD = (54, 284)
IDX_CHEEK = (234, 454)
IDX_JAW = (172, 397)
IDX_LENGTH = (10, 152)
IDX_GONION_LEFT = (172, 234, 152)
IDX_GONION_RIGHT = (397, 454, 152)

RAW_FEATURE_NAMES = ("forehead_width", "cheekbone_width", "jaw_width", "face_length", "jaw_angle_deg")
RATIO_FEATURE_NAMES = ("L_length_over_cheek", "F_forehead_over_cheek", "J_jaw_over_cheek", "A_jaw_angle_deg")

# Prototype per class in (L, F, J, A) standard scores, with the rule it encodes.
PROTOTYPES = {
    # Longer than wide, forehead slightly wider than jaw, soft jaw.
    "oval": (0.5, 0.5, -0.5, 0.5),
    # Short, full jaw, obtuse (soft) jaw angle.
    "round": (-1.0, 0.0, 0.5, 1.0),
    # Short, jaw as wide as the face, angular jaw.
    "square": (-1.0, 0.0, 1.0, -1.0),
    # Long, straight sides, angular jaw.
    "rectangle": (1.5, 0.0, 0.5, -1.0),
    # Wide forehead, narrow pointed jaw.
    "heart": (0.0, 1.5, -1.5, 0.0),
    # Cheekbones widest, narrow forehead and narrow jaw.
    "diamond": (0.5, -1.5, -1.0, 0.0),
    # Jaw wider than forehead.
    "triangle": (0.0, -1.5, 1.5, 0.0),
}
PROTOTYPE_WIDTH = 1.0
SOFTMAX_TEMPERATURE = 1.0


def router_spec() -> dict:
    # Full description of the router, written into every result file.
    return {
        "landmarks": {
            "forehead_width": IDX_FOREHEAD,
            "cheekbone_width": IDX_CHEEK,
            "jaw_width": IDX_JAW,
            "face_length": IDX_LENGTH,
            "jaw_angle_left_vertex_ramus_chin": IDX_GONION_LEFT,
            "jaw_angle_right_vertex_ramus_chin": IDX_GONION_RIGHT,
        },
        "features": RATIO_FEATURE_NAMES,
        "standardisation": "robust: (x - median) / (IQR / 1.349), statistics over all images with landmarks",
        "prototypes": {c: PROTOTYPES[c] for c in config.CLASSES},
        "prototype_width": PROTOTYPE_WIDTH,
        "softmax_temperature": SOFTMAX_TEMPERATURE,
        "trained_on_identity": False,
    }


def _dist(p, a, b):
    return float(np.linalg.norm(p[a] - p[b]))


def _angle(p, vertex, a, b):
    u = p[a] - p[vertex]
    v = p[b] - p[vertex]
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def raw_features(points: np.ndarray) -> np.ndarray:
    # points: (N, 3) landmark array in pixel units.
    angle = 0.5 * (_angle(points, *IDX_GONION_LEFT) + _angle(points, *IDX_GONION_RIGHT))
    return np.array(
        [
            _dist(points, *IDX_FOREHEAD),
            _dist(points, *IDX_CHEEK),
            _dist(points, *IDX_JAW),
            _dist(points, *IDX_LENGTH),
            angle,
        ],
        dtype=np.float64,
    )


def ratio_features(raw: np.ndarray) -> np.ndarray:
    # raw: (n, 5) -> (n, 4) scale-free features (L, F, J, A).
    cheek = raw[:, 1]
    return np.stack([raw[:, 3] / cheek, raw[:, 0] / cheek, raw[:, 2] / cheek, raw[:, 4]], axis=1)


def fit_standardiser(ratios: np.ndarray) -> dict:
    median = np.median(ratios, axis=0)
    q75, q25 = np.percentile(ratios, [75, 25], axis=0)
    scale = (q75 - q25) / 1.349
    return {"median": median.tolist(), "robust_scale": scale.tolist()}


def standardise(ratios: np.ndarray, stats: dict) -> np.ndarray:
    return (ratios - np.asarray(stats["median"])) / np.asarray(stats["robust_scale"])


def shape_probabilities(z: np.ndarray) -> np.ndarray:
    # z: (n, 4) standard scores -> (n, 7) probabilities in config.CLASSES order.
    protos = np.asarray([PROTOTYPES[c] for c in config.CLASSES], dtype=np.float64)
    d2 = (((z[:, None, :] - protos[None, :, :]) / PROTOTYPE_WIDTH) ** 2).sum(axis=2)
    logits = -0.5 * d2 / SOFTMAX_TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


class LandmarkDetector:
    # Wraps MediaPipe Face Landmarker in IMAGE mode. When several faces are
    # found, the one whose landmark centroid is nearest the image centre is
    # used (LFW images are centred on the named person).

    def __init__(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=4,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray):
        # Returns (468, 3) pixel-unit landmarks, or None if no face is found.
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        best, best_d = None, np.inf
        for face in result.face_landmarks:
            pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face[:468]], dtype=np.float64)
            d = np.linalg.norm(pts[:, :2].mean(axis=0) - np.array([w / 2.0, h / 2.0]))
            if d < best_d:
                best, best_d = pts, d
        return best

    def close(self):
        self._landmarker.close()
