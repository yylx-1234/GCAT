"""Train the published GCAT-RP-T model on PASTIS-R or Brandenburg 2018."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.runner import train
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(train(load_config(args.config)))


if __name__ == "__main__":
    main()
