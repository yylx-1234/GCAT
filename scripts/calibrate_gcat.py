"""Fit Validation temperatures and select a development-only decision policy."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.calibrate import calibrate
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True, choices=(
        "primary-fold2", "global-src", "brandenburg-2018"))
    parser.add_argument("--target-risk", type=float, default=0.05)
    args = parser.parse_args()
    report = calibrate(load_config(args.config), args.checkpoint, args.output,
                       args.protocol, args.target_risk)
    print("selected_policy:", report["selected_policy"])


if __name__ == "__main__":
    main()
