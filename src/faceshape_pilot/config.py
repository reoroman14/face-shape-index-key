# Shared constants for the face shape pilot study (see PILOT.md).
# Every result file records SEED and the values defined here.

from pathlib import Path

# Fixed random seed for pair sampling, probe/gallery split, k-means, bootstrap
# and the linear-probe train/test split. Change it only deliberately, and
# re-run everything if you do.
SEED = 20261017

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results" / "pilot"
CACHE_DIR = RESULTS_DIR / "cache"

# Seven classes, in the order used by main.tex.
CLASSES = ("oval", "round", "square", "rectangle", "heart", "diamond", "triangle")

# LFW restriction from PILOT.md.
MIN_FACES_PER_PERSON = 2

# Measurement 1: cap on within-identity pairs per identity.
MAX_PAIRS_PER_IDENTITY = 10

# Bootstrap resamples (over identities) for confidence intervals.
N_BOOTSTRAP = 1000

# Measurement 3: number of k-means buckets equals the number of shape classes.
N_BUCKETS = len(CLASSES)

# Measurement 3 linear probe: fraction of identities held out for testing.
PROBE_TEST_FRACTION = 0.3

# MediaPipe Face Landmarker model (version pinned, not "latest").
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MEDIAPIPE_MODEL_PATH = MODELS_DIR / "mediapipe" / "face_landmarker.task"

# InsightFace model pack: SCRFD detector + ArcFace R50 (WebFace600K).
INSIGHTFACE_PACK = "buffalo_l"
INSIGHTFACE_ROOT = MODELS_DIR / "insightface"

DATASET_LIMITATIONS = (
    "LFW is demographically skewed (predominantly male, lighter-skinned, "
    "middle-aged public figures) and consists of news photographs of public "
    "figures. This is acceptable for a pilot that asks whether the routing "
    "mechanism works at all. It is not acceptable for the fairness analysis "
    "in the paper, which requires a demographically balanced set."
)

# Stage 2 (PILOT2.md).
STAGE2_DIR = REPO_ROOT / "results" / "stage2"
STAGE2_CACHE = STAGE2_DIR / "cache"
STEP0_CPU_REFERENCE = STAGE2_DIR / "step0_cpu_reference"
KAGGLE_DATASET = "niten19/face-shape-dataset"
KAGGLE_DIR = DATA_DIR / "kaggle_faceshape"
KAGGLE_CLASSES = ("heart", "oblong", "oval", "round", "square")

# Stage 3 (PILOT3.md), Part A: FairFace. Pinned Hugging Face revision of
# HuggingFaceM4/FairFace, padding 1.25 (face crop with 1.25 margin, 448 px).
STAGE3_DIR = REPO_ROOT / "results" / "stage3"
STAGE3_CACHE = STAGE3_DIR / "cache"
FAIRFACE_DIR = DATA_DIR / "fairface"
FAIRFACE_HF_REPO = "HuggingFaceM4/FairFace"
FAIRFACE_HF_REVISION = "54d573cdb8b5af490ba8da9da2799628f6e5c496"
FAIRFACE_PADDING = "1.25"
FAIRFACE_RACES = ("East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian")
FAIRFACE_GENDERS = ("Male", "Female")
FAIRFACE_AGES = ("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70")

# Stage 4 (PILOT4.md), Part B: RFW per-group stability.
# RFW test set, loosely cropped .jpg images, obtained from the authors.
# Images derive from MS-Celeb-1M, which was withdrawn by Microsoft in 2019;
# access granted in writing by Mei Wang on 2026-10-20.
STAGE4_DIR = REPO_ROOT / "results" / "stage4"
STAGE4_CACHE = STAGE4_DIR / "cache"
RFW_DIR = DATA_DIR / "rfw" / "images"
RFW_GROUPS = ("Caucasian", "Asian", "Indian", "African")
RFW_EXPECTED = {
    "Caucasian": (10196, 2958),
    "Asian": (9688, 2492),
    "Indian": (10308, 2984),
    "African": (10415, 2995),
}
