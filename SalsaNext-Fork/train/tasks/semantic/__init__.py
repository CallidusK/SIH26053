from pathlib import Path
import sys


# Resolve paths from this file instead of from the caller's working directory.
# This lets inference, evaluation, and visualization run from the repository
# root or from another directory.
TRAIN_PATH = str(Path(__file__).resolve().parents[2])
DEPLOY_PATH = str(Path(__file__).resolve().parents[3] / "deploy")
if TRAIN_PATH not in sys.path:
    sys.path.insert(0, TRAIN_PATH)
