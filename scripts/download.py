#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm53flash.assembler import assemble

parser = argparse.ArgumentParser()
parser.add_argument("destination", nargs="?", type=Path, default=Path("/Users/kumargaurav/Documents/livglm/GLM53Flash"))
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()
assemble(args.destination, workers=args.workers)
