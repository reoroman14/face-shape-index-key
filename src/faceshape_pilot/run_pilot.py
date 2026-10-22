# Runs the whole pilot: extraction (cached), the three measurements and the
# verdict, writing everything to results/pilot/.
#
# Run:  python -m faceshape_pilot.run_pilot [--force-extract]

import argparse

from faceshape_pilot import config, extract, m1_consistency, m2_distribution, m3_partition, verdict
from faceshape_pilot.common import AnalysisSet, environment_info, read_json, write_json
from faceshape_pilot.models import model_hashes


def main(force_extract: bool = False):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    extract.main(force=force_extract)
    ds = AnalysisSet()
    m1_consistency.main(ds)
    m2_distribution.main(ds)
    m3_partition.main(ds)
    verdict.main()
    write_json(config.RESULTS_DIR / "run_manifest.json", {
        "seed": config.SEED,
        "command": "python -m faceshape_pilot.run_pilot",
        "data_accounting": ds.accounting,
        "model_sha256": model_hashes(),
        "mediapipe_model_url": config.MEDIAPIPE_MODEL_URL,
        "insightface_pack": config.INSIGHTFACE_PACK,
        "embedder_runtime": read_json(config.CACHE_DIR / "embed_meta.json"),
        "environment": environment_info(),
        "outputs": sorted(p.name for p in config.RESULTS_DIR.iterdir() if p.is_file()),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-extract", action="store_true", help="recompute router and embedding caches")
    main(ap.parse_args().force_extract)
