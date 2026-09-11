from pathlib import Path
import sys


TRAIN_PATH = str(Path(__file__).resolve().parents[3])
DEPLOY_PATH = str(Path(__file__).resolve().parents[4] / "deploy")
if TRAIN_PATH not in sys.path:
    sys.path.insert(0, TRAIN_PATH)
