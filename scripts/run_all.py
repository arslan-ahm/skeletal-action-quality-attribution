"""Run the whole experiment matrix in the order the results are read in.

    python scripts/run_all.py

Order is not arbitrary. The seed study runs *before* the ablations because an
ablation delta cannot be interpreted without the run-to-run noise scale, and
running it afterwards invites the temptation to read the deltas first.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# The scale is set by the machine, which has four CPU cores shared with other
# jobs, and by a hard ceiling of 20 minutes on any single training run. The
# ablations use a shorter schedule than the headline runs so that nine variants
# fit; that difference is stated in docs/RESULTS.md and the ablation table
# carries its own `full` reference row, so it is internally consistent.
MAIN = ["data.num_sequences=1000", "optim.epochs=10"]
ABLATE = ["data.num_sequences=1000", "optim.epochs=8"]

STAGES: list[tuple[str, list[str]]] = [
    ("compare", ["scripts/compare_methods.py", "--config", "configs/base.yaml",
                 "--set", *MAIN]),
    ("seeds", ["scripts/run_ablations.py", "--only", "seeds", "--set", *MAIN]),
    ("ablations", ["scripts/run_ablations.py", "--only", "ablations",
                   "--set", *ABLATE]),
    ("splits", ["scripts/run_ablations.py", "--only", "splits", "--set", *MAIN]),
    ("data_efficiency", ["scripts/data_efficiency.py", "--set", *MAIN]),
]


def main() -> int:
    for name, argv in STAGES:
        print(f"\n{'=' * 70}\n=== stage: {name}\n{'=' * 70}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run([PY, *argv], cwd=ROOT, check=False)
        print(f"=== stage {name} finished in {(time.perf_counter() - t0) / 60:.1f} min "
              f"(exit {proc.returncode})", flush=True)
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
