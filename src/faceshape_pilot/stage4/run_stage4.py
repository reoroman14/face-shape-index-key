# Runs stage 4 Part B end to end (PILOT4.md) and writes results/stage4/.
#
#   --step0          landmark detection on the first 200 images per group,
#                    printed next to the LFW rate; nothing else runs. The
#                    full run refuses to start until step0.json exists, and
#                    the step 0 rates are to be reviewed before it is started.
#   --manifest-only  write run_manifest.json from existing results.
#
# Run:  python -m faceshape_pilot.stage4.run_stage4 [--step0 | --manifest-only]

import argparse
import subprocess
import sys

from faceshape_pilot import config
from faceshape_pilot.common import environment_info, read_json, write_json
from faceshape_pilot.models import model_hashes
from faceshape_pilot.stage4 import extract_rfw, rfw

STEPS = (
    "faceshape_pilot.stage4.extract_rfw",
    "faceshape_pilot.stage4.part_b",
    "faceshape_pilot.stage4.verdict",
)


def main():
    if not extract_rfw.STEP0_JSON.exists():
        raise SystemExit("step0.json missing: run with --step0 and review the detection rates first")
    config.STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    for module in STEPS:
        subprocess.run([sys.executable, "-m", module], check=True)
    write_manifest()


def write_manifest():
    meta = read_json(extract_rfw.META)
    s0 = read_json(extract_rfw.STEP0_JSON)
    write_json(config.STAGE4_DIR / "run_manifest.json", {
        "seed": config.SEED,
        "command": "python -m faceshape_pilot.stage4.run_stage4",
        "steps": STEPS,
        "provenance": rfw.provenance(),
        "rfw_dir": config.RFW_DIR.relative_to(config.REPO_ROOT).as_posix(),
        "rfw_expected": config.RFW_EXPECTED,
        "rfw_counts_verified": meta["rfw_counts_verified"],
        "step0": {"groups": {g: d["detection_rate"] for g, d in s0["groups"].items()},
                  "lfw_reference": s0["lfw_reference"]},
        "router_seconds": meta["seconds"],
        "router_runtime": "MediaPipe Face Landmarker, CPU (TFLite XNNPACK); ArcFace not run in stage 4",
        "model_sha256": model_hashes(),
        "environment": environment_info(),
        "outputs": sorted(p.name for p in config.STAGE4_DIR.iterdir() if p.is_file()),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step0", action="store_true", help="detection rate on 200 images per group, then stop")
    ap.add_argument("--manifest-only", action="store_true", help="write run_manifest.json from existing results")
    args = ap.parse_args()
    if args.step0:
        extract_rfw.step0()
    elif args.manifest_only:
        write_manifest()
    else:
        main()
