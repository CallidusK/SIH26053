import numpy as np

# 4-class taxonomy constants
TERRAIN = 0
DRIVABLE = 1
STATIC = 2
OBJECT = 3

# Pre-allocated lookup table of size 260 (max class ID is 259)
# By default, everything maps to TERRAIN (0)
_REMAP_LUT = np.zeros(260, dtype=np.uint8)

# 1. DRIVABLE
_REMAP_LUT[[40, 44, 60]] = DRIVABLE

# 2. STATIC
_REMAP_LUT[[50, 51, 52, 70, 71, 80, 81]] = STATIC

# 3. OBJECT
_REMAP_LUT[[
    10, 11, 13, 15, 16, 18, 20, 30, 31, 32,
    252, 253, 254, 255, 256, 257, 258, 259
]] = OBJECT

def map_to_4_classes(pred: np.ndarray) -> np.ndarray:
    """
    Rapidly maps SemanticKITTI class IDs to 4 overarching classes
    using a pre-computed lookup table.
    
    Args:
        pred: np.ndarray of shape (N,) containing raw SemanticKITTI IDs.
    Returns:
        np.ndarray of shape (N,) containing mapped 0-3 IDs.
    """
    return _REMAP_LUT[pred]
