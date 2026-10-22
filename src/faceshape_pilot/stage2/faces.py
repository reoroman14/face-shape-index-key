# Face detection and the CNN input crop, shared by Kaggle and LFW so both
# corpora reach the CNN through the same preprocessing.
#
# Detector: SCRFD from InsightFace buffalo_l (the stage 1 detector, 320x320
# input, CUDA). The face nearest the image centre is used.
#
# Crop rule (fixed before any training):
#   centre = SCRFD box centre, moved up by CROP_UP_SHIFT x box height so the
#            forehead is included (face shape depends on it)
#   side   = CROP_SCALE x max(box width, box height), square
#   output = CROP_SIZE x CROP_SIZE, pixels outside the image replicated from
#            the border
# Optional jitter (Step 3) shifts the centre by (dx, dy) x side and scales
# the side by (1 + ds).
#
# Kaggle images are downscaled to a longest side of KAGGLE_MAX_SIDE on load
# (some are 6,000 to 10,000 pixels). LFW images (250 x 250) are not resized.

import cv2
import numpy as np

CROP_SCALE = 1.5
CROP_UP_SHIFT = 0.10
CROP_SIZE = 224
KAGGLE_MAX_SIDE = 800


def limit_side(rgb: np.ndarray, max_side: int = KAGGLE_MAX_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return rgb
    f = max_side / s
    return cv2.resize(rgb, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)


class FaceDetector:
    def __init__(self):
        from faceshape_pilot.embed import ArcFaceEmbedder

        self._emb = ArcFaceEmbedder(require_cuda=True)

    def providers_in_use(self) -> dict:
        return self._emb.providers_in_use()

    def detect(self, rgb: np.ndarray):
        # Returns (bbox[4], det_score) in pixel coordinates, or (None, None).
        face = self._emb.detect(np.ascontiguousarray(rgb[:, :, ::-1]))
        if face is None:
            return None, None
        return np.asarray(face.bbox, dtype=np.float64), float(face.det_score)


def crop_square(bbox: np.ndarray) -> np.ndarray:
    # Returns (cx, cy, side) of the CNN crop for an SCRFD box.
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0 - CROP_UP_SHIFT * h, CROP_SCALE * max(w, h)])


def extract_crop(rgb: np.ndarray, square: np.ndarray, jitter=(0.0, 0.0, 0.0)) -> np.ndarray:
    # Affine map from the (optionally jittered) square to CROP_SIZE pixels.
    cx, cy, side = square
    dx, dy, ds = jitter
    side = side * (1.0 + ds)
    cx, cy = cx + dx * side, cy + dy * side
    s = CROP_SIZE / side
    m = np.array([[s, 0.0, CROP_SIZE / 2.0 - s * cx], [0.0, s, CROP_SIZE / 2.0 - s * cy]])
    return cv2.warpAffine(rgb, m, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
