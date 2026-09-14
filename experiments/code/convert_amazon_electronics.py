"""Convert Amazon Electronics 5-core reviews to a RecBole .inter file.

The source file is JSON Lines compressed with gzip. The conversion keeps
all interactions and maps Amazon's fields to RecBole atomic-file fields.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


HEADER = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.target.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with gzip.open(args.source, "rt", encoding="utf-8") as source, args.target.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        target.write(HEADER)
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            try:
                user_id = str(record["reviewerID"])
                item_id = str(record["asin"])
                rating = float(record["overall"])
                timestamp = int(record["unixReviewTime"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid Amazon record at source line {line_number}"
                ) from exc

            target.write(f"{user_id}\t{item_id}\t{rating:g}\t{timestamp}\n")
            count += 1

    print(f"Wrote {count:,} interactions to {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
