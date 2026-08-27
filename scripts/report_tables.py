"""Render every committed CSV as a Markdown table.

    python scripts/report_tables.py            # all of them
    python scripts/report_tables.py --only method efficiency

Every table quoted in README.md and docs/RESULTS.md comes from this script, so a
documented number cannot drift from the artefact it came from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saqa.report import ALL, render_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(ALL), default=None)
    args = parser.parse_args(argv)

    if not args.only:
        print(render_all())
        return 0
    for name in args.only:
        print(f"### {name}\n\n{ALL[name]()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
