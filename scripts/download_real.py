"""Retrieval instructions and a loader for real skeleton/AQA datasets.

    python scripts/download_real.py            # print the retrieval steps
    python scripts/download_real.py --check    # report what is present locally

**Nothing in results/ was produced with real data.** Every committed number comes
from the procedural generator, because the real archives below all require
registration or a manual agreement and a repository that cannot be verified
without a login is not reproducible. This script exists so the real path is
first-class when a reader does have the data, and so the format the loader
expects is documented rather than guessed.

Supported layouts, all under ``data/raw/``:

``ntu/`` -- NTU RGB+D 60/120 ``.skeleton`` files (Shahroudy et al., 2016). 25
joints; :func:`saqa.data.real.ntu_to_saqa_topology` maps them onto this project's
17-joint tree. NTU carries *action* labels, not quality scores, so it supports
the architecture and the attribution machinery but not the regression target
without an additional annotation.

``mtl_aqa/`` -- MTL-AQA (Parmar & Morris, 2019): diving clips with real judge
scores, the closest public match to this project's task. Pose must be extracted
first; the loader expects ``poses.npy`` of shape ``(N, T, V, 3)`` and
``scores.csv`` with columns ``clip_id,score``.

``finediving/`` -- FineDiving (Xu et al., 2022), same expected layout, with the
addition of ``phases.csv`` giving the annotated step boundaries -- which is the
one public dataset that can validate the *temporal* half of this project's
attribution claim against human annotation rather than against a generator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

INSTRUCTIONS = """
NTU RGB+D 60/120
  1. Request access at https://rose1.ntu.edu.sg/dataset/actionRecognition/
  2. Download the "3D skeletons" archive (nturgbd_skeletons_s001_to_s017.zip).
  3. Extract into data/raw/ntu/ so that data/raw/ntu/*.skeleton exists.

MTL-AQA (diving, with judge scores)
  1. https://github.com/ParitoshParmar/MTL-AQA -- follow the download link there.
  2. Extract pose with any 3D pose estimator you trust, save as
     data/raw/mtl_aqa/poses.npy  with shape (N, T, V, 3)
     data/raw/mtl_aqa/scores.csv with columns clip_id,score

FineDiving
  1. https://github.com/xujinglin/FineDiving -- request access via the form.
  2. Same layout as MTL-AQA, plus data/raw/finediving/phases.csv
     with columns clip_id,step,start_frame,end_frame

Then:  python scripts/train.py --config configs/real_ntu.yaml
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        root = Path("data/raw")
        found = False
        for name in ("ntu", "mtl_aqa", "finediving"):
            d = root / name
            n = len(list(d.rglob("*"))) if d.exists() else 0
            print(f"  data/raw/{name:<12s} {'present' if n else 'absent':<8s} ({n} files)")
            found = found or bool(n)
        if not found:
            print("\nNo real data found. The default synthetic path needs none.")
        return 0
    print(INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
