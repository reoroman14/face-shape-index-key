# Runs stage 2 end to end (PILOT2.md) and writes results/stage2/.
# Step 0 assumes stage 1 has been re-run on the GPU
# (python -m faceshape_pilot.run_pilot --force-extract) after the CPU
# reference was archived in results/stage2/step0_cpu_reference/.
#
# Run:  python -m faceshape_pilot.stage2.run_stage2 [--manifest-only]

import argparse
import subprocess
import sys

from faceshape_pilot import config
from faceshape_pilot.common import environment_info, read_json, write_json
from faceshape_pilot.models import model_hashes

STEPS = (
    "faceshape_pilot.stage2.step0_check",
    "faceshape_pilot.stage2.kaggle",
    "faceshape_pilot.stage2.step1_train_cnn",
    "faceshape_pilot.stage2.step2_lfw_stability",
    "faceshape_pilot.stage2.step3_perturbation",
    "faceshape_pilot.stage2.verdict",
)


def main():
    config.STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    # Each step in its own process, so no model or ONNX session from one
    # step occupies GPU memory during the next.
    for module in STEPS:
        subprocess.run([sys.executable, "-m", module], check=True)
    write_manifest()


def write_manifest():
    step0 = read_json(config.STAGE2_DIR / "step0_gpu_check.json")
    step1 = read_json(config.STAGE2_DIR / "cnn_training.json")
    write_json(config.STAGE2_DIR / "run_manifest.json", {
        "seed": config.SEED,
        "command": "python -m faceshape_pilot.stage2.run_stage2",
        "steps": STEPS,
        "onnx_providers_in_use": {
            "stage1_arcface_extraction": step0["providers_in_use_gpu_run"],
            "stage2_scrfd_detection": step1["preprocessing"]["detector_providers_in_use"],
        },
        "cnn_selected_backbone": step1["selected_backbone"],
        "model_sha256": model_hashes(),
        "kaggle_archive_sha256": read_json(config.STAGE2_DIR / "kaggle_integrity.json")["archive_sha256"],
        "environment": environment_info(),
        "outputs": sorted(p.name for p in config.STAGE2_DIR.iterdir() if p.is_file()),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-only", action="store_true", help="write run_manifest.json from existing results")
    if ap.parse_args().manifest_only:
        write_manifest()
    else:
        main()
