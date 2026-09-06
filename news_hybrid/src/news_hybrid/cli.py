"""JSON-in/JSON-out command-line adapter."""

import argparse
import json
from pathlib import Path

from .service import decide


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="request JSON")
    parser.add_argument("--state", type=Path, required=True, help="persistent SQLite state")
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(decide(payload, state_path=args.state), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
