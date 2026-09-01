#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm53flash.validator import validate

parser = argparse.ArgumentParser()
parser.add_argument("destination", nargs="?", type=Path, default=Path("/Users/kumargaurav/Documents/livglm/GLM53Flash"))
parser.add_argument("--no-full-hash", action="store_true")
args = parser.parse_args()
validate(args.destination, full_hash=not args.no_full_hash)
