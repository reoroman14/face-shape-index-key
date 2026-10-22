# RFW enumeration, verification and the lightweight analysis set
# (PILOT4.md, Part B).
#
# Images are enumerated by walking data/rfw/images/data/<Group>/<identityID>/,
# not from the *_images.txt lists, and the counts are checked against the
# official figures in config.RFW_EXPECTED; any mismatch is an error.
# *_people.txt is not read: per the RFW readme it lists the identities that
# overlap MS-Celeb-1M, not the full roster.
#
# Identity folders that contain no .jpg at all are not identities and are
# skipped, but they are recorded by name in counts(); the downloaded tree has
# one (Caucasian/m.0g5ty_, empty, absent from Caucasian_images.txt).
#
# RFWSet mirrors only the part of common.AnalysisSet that measurement 1 uses
# (.probs, .top1, .identity, .n_images, .accounting, .images_of()). No
# embeddings are loaded or computed.
#
# Run:  python -m faceshape_pilot.stage4.rfw   (enumerate and verify only)

import re
from dataclasses import dataclass

import numpy as np

from faceshape_pilot import config
from faceshape_pilot.common import AnalysisSet
from faceshape_pilot.models import sha256_file

README = config.DATA_DIR / "rfw" / "readme.txt"
DATASET = ("RFW test set (Wang et al., ICCV 2019), four subsets Caucasian / Asian / Indian / African, "
           "loosely cropped 400x400 .jpg, all identities (every identity has >= 2 images)")
DATASET_LIMITATIONS = (
    "RFW group labels are the subset an identity was placed in by the dataset authors, not "
    "self-identification. Images are selected from MS-Celeb-1M (celebrity web images), so the population "
    "is public figures and the images are uncontrolled web photographs."
)


def provenance() -> dict:
    # Written into every stage 4 result file.
    return {
        "source_dataset": "RFW (Racial Faces in-the-Wild), test subsets only",
        "derived_from": "MS-Celeb-1M, withdrawn by its creator (Microsoft) in April 2019",
        "access": "granted in writing by Mei Wang (Beijing University of Posts and Telecommunications) on "
                  "2026-10-20, in response to a request from Applied Science Private University",
        "obtained_from": "the Google Drive link supplied by the maintainer, not a mirror",
        "downloaded": "the four test subsets only (Caucasian, Asian, Indian, African); the MS1M-wo-RFW and BUPT "
                      "training sets distributed alongside them were not downloaded",
        "verification": "unpacked tree verified file by file against the official counts (40,607 images, "
                        "11,429 identities) before use; re-verified by faceshape_pilot.stage4.rfw on every run",
        "maintainer_readme": README.relative_to(config.REPO_ROOT).as_posix(),
        "maintainer_readme_sha256": sha256_file(README) if README.exists() else None,
    }


@dataclass(frozen=True)
class RFWIndex:
    # Absolute paths, one per image, sorted by group (config.RFW_GROUPS order),
    # then identity, then file name.
    paths: tuple
    # Index into config.RFW_GROUPS per image.
    group: np.ndarray
    # Integer identity id per image (index into names), unique across groups.
    identity: np.ndarray
    # "<Group>/<identityID>", sorted within group.
    names: tuple
    # Group of each identity.
    identity_group: np.ndarray
    # "<Group>/<folder>" of identity folders that contain no .jpg (skipped).
    empty_folders: tuple = ()

    def counts(self) -> dict:
        out = {}
        for gi, g in enumerate(config.RFW_GROUPS):
            ids = self.identity[self.group == gi]
            per_id = np.bincount(ids, minlength=len(self.names))[self.identity_group == gi]
            out[g] = {"images": int(len(ids)), "identities": int((self.identity_group == gi).sum()),
                      "identities_ge2": int((per_id >= config.MIN_FACES_PER_PERSON).sum()),
                      "empty_folders_skipped": [e for e in self.empty_folders if e.startswith(g + "/")]}
        return out


def load_rfw_index() -> RFWIndex:
    root = config.RFW_DIR / "data"
    if not root.is_dir():
        raise FileNotFoundError(f"Expected RFW folder not found: {root}")

    paths, group, identity, names, identity_group = [], [], [], [], []
    bad_names, empty = [], []
    for gi, g in enumerate(config.RFW_GROUPS):
        gdir = root / g
        if not gdir.is_dir():
            raise FileNotFoundError(f"Expected RFW group folder not found: {gdir}")
        for person_dir in sorted(p for p in gdir.iterdir() if p.is_dir()):
            files = sorted(person_dir.glob("*.jpg"))
            if not files:
                empty.append(f"{g}/{person_dir.name}")
                continue
            pattern = re.compile(re.escape(person_dir.name) + r"_\d+\.jpg")
            bad_names += [f for f in files if not pattern.fullmatch(f.name)]
            ident = len(names)
            names.append(f"{g}/{person_dir.name}")
            identity_group.append(gi)
            for f in files:
                paths.append(str(f))
                group.append(gi)
                identity.append(ident)

    index = RFWIndex(
        paths=tuple(paths),
        group=np.asarray(group, dtype=np.int64),
        identity=np.asarray(identity, dtype=np.int64),
        names=tuple(names),
        identity_group=np.asarray(identity_group, dtype=np.int64),
        empty_folders=tuple(empty),
    )
    if bad_names:
        raise RuntimeError(f"{len(bad_names)} RFW files do not follow <identityID>_<faceID>.jpg, "
                           f"first: {bad_names[0]}")
    counts = index.counts()
    mismatch = {g: {"found": (c["images"], c["identities"], c["identities_ge2"]),
                    "expected": (*config.RFW_EXPECTED[g], config.RFW_EXPECTED[g][1])}
                for g, c in counts.items()
                if (c["images"], c["identities"], c["identities_ge2"])
                != (*config.RFW_EXPECTED[g], config.RFW_EXPECTED[g][1])}
    if mismatch:
        raise RuntimeError(f"RFW folder walk disagrees with the official counts "
                           f"(images, identities, identities with >= 2 images): {mismatch}")
    return index


class RFWSet:
    # Images of one RFW group for which landmark detection succeeded,
    # restricted to identities that still have at least two such images.
    # probs is the (n_all, 7) router output for the whole cache under one
    # standardiser condition; rows of failed images are NaN and never used.

    images_of = AnalysisSet.images_of

    def __init__(self, cache: dict, probs: np.ndarray, group_index: int):
        in_group = cache["group"] == group_index
        ok = cache["landmark_ok"] & in_group
        ident_all = cache["identity"]
        n_names = len(cache["names"])
        counts = np.bincount(ident_all[ok], minlength=n_names)
        keep = ok & (counts[ident_all] >= config.MIN_FACES_PER_PERSON)

        self.keep = keep
        self.group = config.RFW_GROUPS[group_index]
        self.paths = cache["paths"][keep]
        uniq, self.identity = np.unique(ident_all[keep], return_inverse=True)
        self.names = cache["names"][uniq]
        self.probs = probs[keep]
        self.ratio_features = cache["ratio_features"][keep]
        self.top1 = self.probs.argmax(axis=1)
        self.n_images = int(keep.sum())
        self.n_identities = len(uniq)
        ids_in_group = np.unique(ident_all[in_group])
        self.accounting = {
            "group": self.group,
            "rfw_images": int(in_group.sum()),
            "rfw_identities": int(len(ids_in_group)),
            "decode_failed": int((in_group & ~cache["decoded"]).sum()),
            "landmark_failed": int((in_group & cache["decoded"] & ~cache["landmark_ok"]).sum()),
            "dropped_identity_below_2_after_failures_images": int((ok & ~keep).sum()),
            "dropped_identities_below_2": int(len(ids_in_group) - len(uniq)),
            "analysis_images": self.n_images,
            "analysis_identities": self.n_identities,
        }


def main():
    index = load_rfw_index()
    for g, c in index.counts().items():
        print(f"[rfw] {g:10s} {c['images']:6d} images  {c['identities']:5d} identities  "
              f"{c['identities_ge2']:5d} with >= 2 images")
    print(f"[rfw] verified: {len(index.paths)} images, {len(index.names)} identities; "
          f"empty folders skipped: {list(index.empty_folders)}")


if __name__ == "__main__":
    main()
