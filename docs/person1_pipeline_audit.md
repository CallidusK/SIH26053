# Data & Pose Pipeline (Person 1) — Audit & Usage Guide

This document serves as an in-depth audit and integration guide for the rest of the Foveated Semantic 2.5D LiDAR Mapping team. It covers the newly implemented ingestion pipeline, ensuring everyone understands how to use the outputs without violating the architecture's core design principles.

> [!IMPORTANT]
> **To the Team:** This pipeline guarantees two things:
> 1. Absolute zero dependency on Open3D, RANSAC, or Kalman filters.
> 2. End-to-end `NumPy` operation on the hot path. 
>
> Do not attempt to cast these arrays to other formats unless explicitly handing off to the PyTorch segmentation model.

---

## 1. Executive Summary
The Person 1 deliverable (`kitti_loader.py` + `remap.py`) successfully abstracts the messy reality of the SemanticKITTI dataset into a clean, iterable interface. It discovers `.bin` scans automatically, parses calibration/pose text files, safely computes LiDAR-relative motion (`T_rel`), and instantly maps 32-bit semantic class IDs into the 4-class taxonomy using an O(1) lookup table. 

All gates (A1-A5, B1, and REMAP) have passed verification.

---

## 2. Integration Guide (How to use this code)

If you are Persons 2, 3, 4, 5, or 6, this is how you interact with the pipeline:

### Initializing the Loader
```python
from kitti_loader import SemanticKITTILoader

# Point it to the specific sequence directory
loader = SemanticKITTILoader("data/sequences/08")

# You can easily check how many frames exist
print(len(loader)) # Output: 4007
```

### The `points` Array (N, 4) — For Persons 2, 3, and 4
**What it is:** A 2D NumPy array of 32-bit floats (`float32`). `N` is the number of LiDAR points in the current sweep (usually around 17,000 to 20,000 for SemanticKITTI sequence 08). The 4 columns represent `[x, y, z, intensity]`.

**How you use it:** 
- **Persons 2 & 3 (Segmentation):** This is the raw input to your neural network pipeline. You will take this array, project it into a 2D spherical range image (e.g., 64 x 2048), run the SalsaNext model, and output a semantic label for each of the N points.
- **Person 4 (Grid Engine):** Once the points are labeled, you will use the `x` and `y` coordinates from this same array to bucket each point into the correct x, y cell of the 2.5D foveated grid.
- **Rule:** Do *not* apply any spatial transformations to this array! The current scan stays exactly as the sensor saw it (robot-centric).

```python
for points, pose_curr, pose_prev, Tr, frame_id in loader:
    # 'points' is your raw (N, 4) array!
    # Pass it downstream to Segmentation (Persons 2/3)
    pass
```

### The `T_rel` Matrix (4, 4) — For Person 5
**What it is:** A 4 x 4 homogeneous transformation matrix of 64-bit floats (`float64`). It precisely mathematically defines how the car moved *from the previous frame to the current frame*, specifically translated into the LiDAR sensor's coordinate system (extrinsic conjugation).

**How you use it:**
- **Person 5 (Rating & Fusion):** You use this matrix to move the *historical map* into the *current frame*. Because the map is "robot-centric" (always centered around 0,0 where the robot currently is), when the robot moves forward by 1 meter, the entire previously built map must be shifted backward by 1 meter so it aligns with the new scan. 
- **Rule:** `T_rel` is used to re-bin the old `MapSnapshot`. It is **never** applied to the current scan's `points` array.

```python
T_rel = loader.get_T_rel(frame_id)
# Use T_rel to warp the previously stored MapSnapshot into the current frame
# before fusing the new scan data.
```

### Semantic Remapping (For Persons 2/3)
Once the neural network outputs raw SemanticKITTI class IDs, map them to the 4 categories instantly:

```python
from remap import map_to_4_classes

# 'raw_preds' is an array of IDs from 0 to 259
mapped_classes = map_to_4_classes(raw_preds)
# Output is an array of the same shape, containing only 0, 1, 2, or 3.
```

---

## 3. In-Depth Function Audit

This section explains the internal logic of every function written by Person 1 to ensure transparency across the team.

### `kitti_loader.py`

#### `load_point_cloud(bin_path, dtype, num_fields)`
- **What it does:** Reads a raw binary LiDAR sweep.
- **Audit Details:** Uses `np.fromfile` for maximum speed. Hardcoded to reshape into `(-1, 4)` to extract `[x, y, z, intensity]`. Does absolutely no filtering; returns exactly what the sensor saw.

#### `load_poses(poses_path)`
- **What it does:** Reads the sequence's camera trajectory.
- **Audit Details:** Parses the 12-float lines from `poses.txt` into 3 x 4 matrices, then pads them with a bottom row `[0, 0, 0, 1]` to create proper 4 x 4 homogeneous transformation matrices. These are absolute camera poses.

#### `load_calib_tr(calib_path)`
- **What it does:** Extracts the Velodyne-to-Camera extrinsic matrix.
- **Audit Details:** Scans `calib.txt` specifically for the `"Tr:"` line. Like the poses, it pads the 12 floats into a 4 x 4 matrix. Essential for translating the camera poses into LiDAR-centric motion.

#### `compute_T_rel(pose_curr, pose_prev, Tr)`
- **What it does:** Calculates the exact motion of the car from the previous frame to the current frame, *from the perspective of the LiDAR sensor*.
- **Audit Details:** Implements the extrinsic conjugation formula: `inv(Tr) @ inv(pose_curr) @ pose_prev @ Tr`.
  > [!WARNING]
  > **Crucial Architecture Rule:** This matrix (`T_rel`) exists solely to shift the *historical map backward*. The current scan (`points`) is **never** multiplied by `T_rel`.

#### `SemanticKITTILoader.__init__(self, sequence_dir)`
- **What it does:** Scaffolds the dataset sequence.
- **Audit Details:** Does not hardcode file counts. It glob-searches for `.bin` files and asserts that the number of parsed poses perfectly matches the number of `.bin` files. 

#### `SemanticKITTILoader.__getitem__(self, idx)`
- **What it does:** Fetches all data required for a specific frame index.
- **Audit Details:** Safely handles `idx = 0` by returning a 4 x 4 Identity matrix for `pose_prev` (since there is no motion before the first frame).

---

### `remap.py`

#### `map_to_4_classes(pred)`
- **What it does:** Reduces the 260 possible SemanticKITTI classes down to 4 core groups: `TERRAIN (0)`, `DRIVABLE (1)`, `STATIC (2)`, and `OBJECT (3)`.
- **Audit Details:** **Performance Critical.** Instead of using slow, chained boolean masks (`pred == 40 | pred == 44`), it uses a pre-allocated NumPy array (`_REMAP_LUT`) of size 260. Calling `_REMAP_LUT[pred]` performs the mapping in a single, instantaneous memory lookup operation.
- **Class Groupings:**
  - `DRIVABLE`: Roads, parking, and lane markings.
  - `STATIC`: Buildings, fences, poles, trunks, traffic signs, etc.
  - `OBJECT`: All vehicles, people, and their `moving-*` counterparts (classes 252-259).
  - `TERRAIN`: Everything else (sidewalks, vegetation, outlier/unlabeled).
