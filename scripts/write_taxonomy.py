"""Write the error taxonomy beside the data it describes.

The article names its error classes and never prints their codes, so a reader
following the argument is not asked to hold a key in their head. A reader who
opens the released data meets the codes immediately, in every item identifier
and every response file, and needs the key at once. This writes it where that
reader is rather than where the first reader is.

The table is generated from the configuration rather than typed, so a class
renamed in one place cannot disagree with itself in another.

Outputs:
    data/processed/injected_set/taxonomy.csv

Usage:
    python scripts/write_taxonomy.py
"""

from __future__ import annotations

import csv

from poi_audit.config import get, path
from poi_audit.logsetup import setup

logger = setup("write_taxonomy")

# How the configuration's machine words read to a person.
MECHANISMS = {
    "format_rule": "a format rule settles it",
    "reference_lookup": "a reference lookup settles it",
    "none": "neither settles it",
}


def main() -> None:
    """Write the code, the name and the settling mechanism for every class."""
    classes = get("injected_set.classes")
    target = path("data_processed") / "injected_set" / "taxonomy.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "name", "kind", "settled_by", "items"])
        for code in sorted(classes):
            entry = classes[code]
            writer.writerow(
                [
                    code,
                    entry["description"],
                    entry["kind"],
                    MECHANISMS.get(str(entry["settled_by"]), entry["settled_by"]),
                    entry["items"],
                ]
            )
    logger.info(f"wrote {target} with {len(classes)} classes")


if __name__ == "__main__":
    main()
