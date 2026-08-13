from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / ".codex_pydeps"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fpl_predictor.train import train_all


if __name__ == "__main__":
    train_all()
