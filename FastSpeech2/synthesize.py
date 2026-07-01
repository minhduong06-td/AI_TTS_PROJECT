#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

runpy.run_module("fastspeech2.inference.synthesize", run_name="__main__")
