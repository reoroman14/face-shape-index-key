# ArcFace embedding via InsightFace (buffalo_l: SCRFD detector + ArcFace R50
# trained on WebFace600K). One L2-normalised 512-d embedding per image.
#
# Independent of the router: it uses its own detector and its own 5-point
# alignment, and shares no landmarks or code with router.py.
#
# Execution provider: CUDA with CPU fallback. onnxruntime falls back to CPU
# silently when CUDA libraries are missing, so the providers actually bound
# to each session are read back at runtime (providers_in_use) and recorded.
# By default a missing CUDA provider is an error, not a silent slowdown.
#
# Stage 1 used a second detector instance at 640x640 as a fallback. The
# stage 1 cache shows it was used for 0 of 9,164 images, so it was removed.
#
# The buffalo_l weights are released for non-commercial research use only.

import numpy as np

from faceshape_pilot import config

DET_SIZE = (320, 320)
REQUESTED_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class ArcFaceEmbedder:
    def __init__(self, require_cuda: bool = True):
        from insightface.app import FaceAnalysis

        config.INSIGHTFACE_ROOT.mkdir(parents=True, exist_ok=True)
        self.app = FaceAnalysis(
            name=config.INSIGHTFACE_PACK,
            root=str(config.INSIGHTFACE_ROOT),
            allowed_modules=["detection", "recognition"],
            providers=REQUESTED_PROVIDERS,
        )
        self.app.prepare(ctx_id=0, det_size=DET_SIZE)
        in_use = self.providers_in_use()
        if require_cuda and any(p[0] != "CUDAExecutionProvider" for p in in_use.values()):
            raise RuntimeError(f"CUDAExecutionProvider not active; sessions bound to {in_use}")

    def providers_in_use(self) -> dict:
        # Provider list actually bound to each loaded ONNX session.
        return {task: list(m.session.get_providers()) for task, m in self.app.models.items()}

    def detect(self, bgr: np.ndarray):
        # Returns the detected face nearest the image centre, or None.
        faces = self.app.get(bgr)
        if not faces:
            return None
        h, w = bgr.shape[:2]
        centre = np.array([w / 2.0, h / 2.0])
        return min(
            faces,
            key=lambda f: np.linalg.norm(
                np.array([(f.bbox[0] + f.bbox[2]) / 2.0, (f.bbox[1] + f.bbox[3]) / 2.0]) - centre
            ),
        )

    def embed(self, rgb: np.ndarray):
        # Returns (embedding, det_score, det_size_used) or (None, None, None).
        face = self.detect(np.ascontiguousarray(rgb[:, :, ::-1]))
        if face is None:
            return None, None, None
        emb = np.asarray(face.normed_embedding, dtype=np.float32)
        return emb, float(face.det_score), DET_SIZE[0]
