# Runs stage 3 Part A end to end (PILOT3.md) and writes results/stage3/.
# Part B (RFW) is not implemented: access has not been obtained.
#
# Run:  python -m faceshape_pilot.stage3.run_stage3 [--manifest-only]

import argparse
import subprocess
import sys

from faceshape_pilot import config
from faceshape_pilot.common import environment_info, read_json, write_json
from faceshape_pilot.models import model_hashes

STEPS = (
    "faceshape_pilot.stage3.fairface",
    "faceshape_pilot.stage3.extract_fairface",
    "faceshape_pilot.stage3.part_a",
    "faceshape_pilot.stage3.verdict",
)


def main():
    config.STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    for module in STEPS:
        subprocess.run([sys.executable, "-m", module], check=True)
    write_manifest()


def write_manifest():
    dl = read_json(config.STAGE3_DIR / "fairface_download.json")
    meta = read_json(config.STAGE3_CACHE / "fairface_router_meta.json")
    write_json(config.STAGE3_DIR / "run_manifest.json", {
        "seed": config.SEED,
        "command": "python -m faceshape_pilot.stage3.run_stage3",
        "steps": STEPS,
        "part_b": "not run: RFW access not obtained",
        "fairface": {"repo": config.FAIRFACE_HF_REPO, "revision": dl["revision"], "padding": dl["padding"],
                     "shards": dl["shards"]},
        "router_seconds": meta["seconds"],
        "router_runtime": "MediaPipe Face Landmarker, CPU (TFLite XNNPACK); no GPU used in stage 3",
        "model_sha256": model_hashes(),
        "environment": environment_info(),
        "outputs": sorted(p.name for p in config.STAGE3_DIR.iterdir() if p.is_file()),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-only", action="store_true", help="write run_manifest.json from existing results")
    if ap.parse_args().manifest_only:
        write_manifest()
    else:
        main()
