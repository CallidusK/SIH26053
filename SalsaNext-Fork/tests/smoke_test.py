"""Minimal no-data check for the standard SalsaNext model."""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from train.tasks.semantic.modules.SalsaNext import SalsaNext  # noqa: E402


def main():
    model = SalsaNext(nclasses=20).eval()
    inputs = torch.randn(1, 5, 64, 128)

    with torch.no_grad():
        outputs = model(inputs)

    expected_shape = (1, 20, 64, 128)
    assert outputs.shape == expected_shape, (outputs.shape, expected_shape)
    assert torch.allclose(
        outputs.sum(dim=1), torch.ones(1, 64, 128), atol=1e-5
    )
    print("SalsaNext smoke test passed")


if __name__ == "__main__":
    main()
