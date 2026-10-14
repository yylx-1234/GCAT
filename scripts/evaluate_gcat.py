"""Evaluate a frozen GCAT checkpoint and development-selected policy."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.run import evaluate
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(load_config(args.config), args.checkpoint, args.policy, args.output)
    print(report["operational"])


if __name__ == "__main__":
    main()
