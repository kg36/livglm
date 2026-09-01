#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm53flash.preflight import run_preflight

run_preflight(ROOT / "reports" / "preflight.json")
