# FOVMAP + SalsaNext: Integration Guide & Pipeline Reference

A unified guide to integrating the **FOVMAP** 2.5D foveated LiDAR grid mapping pipeline with the **SalsaNext** real-time uncertainty-aware semantic segmentation neural network.

---

## 1. System Overview & Architecture

The objective of this system is to build an online, robot-centric, 2.5D foveated elevation and semantic grid map from 3D LiDAR point clouds at autonomous driving sensor frame rates (10 Hz or higher).

### The Pipeline Workflow

```
Raw KITTI LiDAR (.bin)
         │
         ▼
[fovmap.data.loader] ────────► Raw Point Cloud: (N, 4) [x, y, z, intensity]
         │                     Extrinsics & Poses: T_rel (4, 4)
         │
         ▼
[Spherical Range Projection] ─► Range Image Tensor: (5, 64, 2048) [range, x, y, z, remission]
         │
         ▼
[SalsaNext Neural Network] ──► 20-Class Logits: (20, 64, 2048) + Epistemic Uncertainty
         │
         ▼
[Point-Wise Unprojection] ───► Point-Wise Semantic Classes (0 to 19 or 0 to 259)
         │
         ▼
[fovmap.data.remap] ─────────► 4-Class Taxonomy: (N,) uint8 [TERRAIN, DRIVABLE, STATIC, OBJECT]
         │
         ▼
[2.5D Foveated Grid Engine] ──► Multi-ring Elevation & Occupancy Grid (10m, 25m, 50m rings)
```

### Roles and Boundaries
- **Person 1 (`fovmap.data`)**: Fast ingestion of binary LiDAR scans, camera-to-LiDAR calibration parsing (`Tr`), pose tracking, relative transformation calculation (`T_rel`), and taxonomy mapping (`remap.py`).
- **Persons 2 & 3 (`fovmap.segmentation` / SalsaNext)**: Range-image projection, SalsaNext forward inference, pixel-to-point unprojection, and uncertainty estimation.
- **Person 4 (`fovmap.grid`)**: Spatial aggregation of labeled points into concentric foveated rings (5cm inner, up to 50cm outer).
- **Person 5 (`fovmap.rating`)**: Traversability calculation, dynamic object tagging, and log-odds map fusion using `T_rel`.
- **Person 6 (`fovmap.dashboard`)**: Headless latency benchmark tracking and UI visualization.

---

## 2. Parameterized Sequence Selection (Recent Changes)

Previously, test scripts and verification routines assumed sequence `08` in hardcoded path strings. The verification harness (`verify_gates.py`) has been fully refactored to remove all hardcoded sequence dependencies.

### Key Changes
1. **Configurable Sequence Parameter**:
   All verification gates now accept `seq_id` as a parameter (defaulting to `"04"`):
   - `check_gate_a1(data_root, seq_id="04", ...)`
   - `check_gate_a2(data_root, seq_id="04")`
   - `check_gate_a3(data_root, seq_id="04")`
   - `check_gate_a4(data_root, seq_id="04")`
   - `check_gate_a5(data_root, seq_id="04")`
   - `check_gate_b1(data_root, seq_id="04")`
2. **Command-Line Interface (CLI)**:
   `verify_gates.py` now includes an `argparse` CLI:
   - `--seq` / `-s`: Sequence ID (default: `"04"`).
   - `--data-root` / `-d`: Root directory containing `sequences/` (default: `sample_kitti`).
3. **Graceful Diagnostics**:
   If a sequence directory or file is missing, the script prints an explicit error detailing the missing path and provides actionable guidance (e.g. suggesting `--seq 08` if using the sample data).

### Environment Setup (Virtual Environment)

On macOS and Linux, the system shell may not alias `python` by default, leading to `zsh: command not found: python`. Dependencies (like `numpy`) are installed inside the repository's `.venv`.

You can run commands using either of these two methods:

```bash
# Method 1: Activate the virtual environment (recommended)
source .venv/bin/activate
python verify_gates.py -s 08

# Method 2: Run directly via virtual environment python binary
./.venv/bin/python verify_gates.py -s 08
```

---

### How to Run Verification

#### A. Verifying a Single Sequence via CLI
Use `--seq` / `-s` to choose which sequence to verify, and `--data-root` / `-d` to specify the dataset directory:

```bash
# Verify sequence 08 on the sample dataset
python verify_gates.py -s 08

# Verify sequence 04 on a full SemanticKITTI dataset
python verify_gates.py -d /path/to/SemanticKITTI/dataset -s 04
```

#### B. Verifying a Single .bin Scan Directly
To inspect or debug a single binary point cloud scan without running all gates:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fovmap.data.loader import load_point_cloud

# Read any .bin file directly
bin_file = Path("sample_kitti/sequences/08/velodyne/000000.bin")
points = load_point_cloud(bin_file)

print(f"Total points : {points.shape[0]}")
print(f"Shape        : {points.shape}")      # (N, 4)
print(f"Dtype        : {points.dtype}")      # float32
print(f"x range      : [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] m")
print(f"y range      : [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] m")
print(f"z range      : [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] m")
print(f"Intensity    : [{points[:, 3].min():.2f}, {points[:, 3].max():.2f}]")
```

#### C. Verifying an Entire Sequence of .bin Scans
To iterate through all scans in a sequence, check point counts, and track translation:

```python
from pathlib import Path
from fovmap.data.loader import SemanticKITTILoader

seq_dir = Path("sample_kitti/sequences/08")
loader = SemanticKITTILoader(seq_dir)

print(f"Sequence has {len(loader)} total frames.")

for frame_idx, (points, pose, pose_prev, Tr, frame_id) in enumerate(loader):
    assert points.shape[1] == 4
    assert points.shape[0] > 0
    t = pose[:3, 3]
    print(f"Frame {frame_id:04d}: {points.shape[0]} points, translation = [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] m")
```

#### D. What the Verification Gates Check

| Gate | Target | What is Verified |
|---|---|---|
| **Gate A1** | Single `.bin` scan | 4-float fields (x, y, z, intensity), file byte-size divisibility, finite values, intensity in [0, 1] |
| **Gate A2** | `poses.txt` | (N, 12) parsed into (N, 4, 4) poses, pose[0] is identity, translation steps between 0.1m and 5.0m |
| **Gate A3** | `calib.txt` | Tr parsed to (4, 4), bottom row [0, 0, 0, 1], rotation matrix det(R) ~ 1.0 |
| **Gate A4** | `SemanticKITTILoader[0]` | Frame 0 retrieval, pose_prev is identity, types and shapes correct |
| **Gate A5** | Full Sequence | Iterates all `.bin` files in sequence, verifying non-empty point clouds and clean loading |
| **Gate B1** | Relative Pose `T_rel` | Compares computed relative motion against ground-truth transitions |
| **Gate REMAP** | 4-Class Taxonomy | Verifies O(1) LUT correctly maps 260 SemanticKITTI classes to 4 target classes |

---

## 3. Interfacing with SalsaNext

SalsaNext operates on 2D cylindrical range images rather than raw unorganized 3D point clouds. This section explains how to format inputs and handle outputs.

### 3.1 Input Tensor Format: (5, H, W)
SalsaNext expects a 5-channel range image:
- **Channel 0**: Range (depth) `r = sqrt(x^2 + y^2 + z^2)`
- **Channel 1**: `x` coordinate
- **Channel 2**: `y` coordinate
- **Channel 3**: `z` coordinate
- **Channel 4**: Remission (LiDAR intensity normalized between 0.0 and 1.0)

Typical grid dimensions for Velodyne HDL-64E:
- Height `H = 64` (number of laser vertical beams)
- Width `W = 2048` (azimuthal horizontal resolution)

### 3.2 Spherical Projection Geometry (Plain Text)

To project a point `P = [x, y, z]` into spherical coordinates:
- Range: `r = sqrt(x^2 + y^2 + z^2)`
- Azimuth (yaw): `yaw = -atan2(y, x)`
- Elevation (pitch): `pitch = arcsin(z / r)`

Given sensor field of view:
- `fov_up = 3.0 degrees` (+0.0524 rad)
- `fov_down = -25.0 degrees` (-0.4363 rad)
- Total vertical FOV: `fov = fov_up - fov_down = 28.0 degrees` (+0.4887 rad)

Calculating image coordinates `(u, v)`:
- Column index `u`:
  `u = 0.5 * (1.0 - yaw / pi) * W`
  `u = clip(floor(u), 0, W - 1)`
- Row index `v`:
  `v = (1.0 - (pitch + abs(fov_down)) / fov) * H`
  `v = clip(floor(v), 0, H - 1)`

---

## 4. Semantic Label Remapping (O(1) LUT)

SalsaNext outputs predictions over 20 learning classes (from SemanticKITTI classes 0 to 259). For 2.5D foveated elevation and occupancy mapping, the system compresses these into a 4-class taxonomy.

### The 4-Class Taxonomy

| Class ID | Class Name | Included SemanticKITTI Classes | Mapping Purpose |
|---|---|---|---|
| `0` | **TERRAIN** | Unlabeled (0), Outlier (1), Sidewalk (48), Terrain (72), Vegetation (70) | Ground cells that are not designated for driving |
| `1` | **DRIVABLE** | Road (40), Parking (44), Other-ground (49), Lane-marking (60) | Surfaces safely traversable by the vehicle |
| `2` | **STATIC** | Building (50), Fence (51), Pole (80), Traffic-sign (81), Trunk (71) | Rigid non-traversable architectural obstacles |
| `3` | **OBJECT** | Car (10), Bicycle (11), Motorcycle (15), Truck (18), Person (30), Moving-* (252-259) | Dynamic or movable actors requiring change detection |

### Using `fovmap.data.remap`

The remapping uses a pre-allocated lookup table of 260 entries (`_REMAP_LUT` in `src/fovmap/data/remap.py`):

```python
import numpy as np
from fovmap.data.remap import map_to_4_classes

# Example: Predictions from model unprojection
raw_predictions = np.array([40, 50, 10, 72, 252], dtype=np.uint32)

# Fast vectorized O(1) lookup
mapped = map_to_4_classes(raw_predictions)

# Result: array([1, 2, 3, 0, 3], dtype=uint8)
print(mapped)
```

Benefits:
- Eliminates chained boolean masks (`pred == 40 | pred == 44`).
- Constant-time O(1) memory lookup.
- Zero allocations on the per-scan hot path.

---

## 5. Relative Pose (`T_rel`) & Temporal Alignment

Autonomous driving mapping must remain robot-centric to avoid coordinate drift over multi-kilometer sequences.

### Mathematical Formulation
The car's motion between consecutive frames is defined from the perspective of the LiDAR sensor via extrinsic conjugation:

```
T_rel = inv(Tr) @ inv(pose_curr) @ pose_prev @ Tr
```

Where:
- `pose_curr` is the (4, 4) absolute camera pose at time `t`.
- `pose_prev` is the (4, 4) absolute camera pose at time `t - 1`.
- `Tr` is the (4, 4) calibration matrix mapping Velodyne coordinates to Camera coordinates (`X_cam = Tr @ X_velo`).
- `@` denotes standard matrix multiplication.

### The Cardinal Golden Rule
> [!IMPORTANT]
> **The current scan's point cloud is NEVER transformed by `T_rel`.**
> The current scan remains exactly as captured in the sensor frame (robot-centric at `[0, 0, 0]`).
> `T_rel` is applied exclusively to warp the **historical map snapshot** backward by the ego-vehicle's movement before fusing new information.

### Multi-Frame Temporal SalsaNext Integration
When running temporal variants of SalsaNext (which take scans from `t` and `t - 1`):
1. Load scan `t` using `loader[idx]`.
2. Load scan `t - 1` using `loader[idx - 1]`.
3. Compute `T_rel = loader.get_T_rel(idx)`.
4. Transform points from `t - 1` into the current frame `t`:
   `points_prev_in_curr = (T_rel @ points_prev_homogeneous.T).T`
5. Project both clouds into their respective range images and stack them as multi-temporal channel inputs.

---

## 6. End-to-End Integration Recipe

The following standalone script demonstrates loading a frame, generating a 5-channel range image for SalsaNext, unprojecting predictions, and applying the 4-class taxonomy:

```python
import sys
from pathlib import Path
import numpy as np

# Ensure fovmap is discoverable if running from workspace root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fovmap.data.loader import SemanticKITTILoader
from fovmap.data.remap import map_to_4_classes

def project_range_image(points, H=64, W=2048, fov_up=3.0, fov_down=-25.0):
    """
    Project an (N, 4) point cloud [x, y, z, intensity] into a (5, H, W) range image.
    Returns:
        proj_tensor: (5, H, W) float32 numpy array
        proj_idx: (H, W) int32 point index map (-1 for empty pixels)
    """
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    depth = np.linalg.norm(points[:, :3], axis=1)

    # Filter invalid points
    valid = depth > 0.5
    x, y, z, intensity, depth = x[valid], y[valid], z[valid], intensity[valid], depth[valid]
    orig_indices = np.where(valid)[0]

    # Spherical coordinates
    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(z / depth)

    fov_up_rad = np.deg2rad(fov_up)
    fov_down_rad = np.deg2rad(fov_down)
    total_fov = fov_up_rad - fov_down_rad

    # Image projection indices
    u = 0.5 * (1.0 - yaw / np.pi) * W
    u = np.clip(np.floor(u).astype(np.int32), 0, W - 1)

    v = (1.0 - (pitch + abs(fov_down_rad)) / total_fov) * H
    v = np.clip(np.floor(v).astype(np.int32), 0, H - 1)

    # Fill range tensor (sorting by depth descending so nearer points overwrite)
    order = np.argsort(-depth)
    u_sorted = u[order]
    v_sorted = v[order]
    depth_sorted = depth[order]
    x_sorted = x[order]
    y_sorted = y[order]
    z_sorted = z[order]
    intensity_sorted = intensity[order]
    orig_idx_sorted = orig_indices[order]

    proj_tensor = np.zeros((5, H, W), dtype=np.float32)
    proj_idx = np.full((H, W), -1, dtype=np.int32)

    proj_tensor[0, v_sorted, u_sorted] = depth_sorted
    proj_tensor[1, v_sorted, u_sorted] = x_sorted
    proj_tensor[2, v_sorted, u_sorted] = y_sorted
    proj_tensor[3, v_sorted, u_sorted] = z_sorted
    proj_tensor[4, v_sorted, u_sorted] = intensity_sorted
    proj_idx[v_sorted, u_sorted] = orig_idx_sorted

    return proj_tensor, proj_idx

# --- Usage Pipeline ---
if __name__ == "__main__":
    data_dir = Path("sample_kitti/sequences/08")
    loader = SemanticKITTILoader(data_dir)

    # 1. Fetch current frame
    points, pose_curr, pose_prev, Tr, frame_id = loader[0]
    T_rel = loader.get_T_rel(frame_id)

    print(f"Loaded frame {frame_id}: {points.shape[0]} points")
    print(f"T_rel translation vector: {T_rel[:3, 3]}")

    # 2. Build SalsaNext range image
    range_img, point_indices = project_range_image(points)
    print(f"Range tensor shape: {range_img.shape}")

    # 3. Simulate SalsaNext predictions (e.g. shape H, W)
    # In practice: preds = salsanext_model(torch.from_numpy(range_img).unsqueeze(0))
    dummy_preds = np.random.choice([40, 50, 10, 72], size=(64, 2048))

    # 4. Unproject image labels back to original point cloud
    point_labels = np.zeros(points.shape[0], dtype=np.uint32)
    valid_mask = point_indices >= 0
    point_labels[point_indices[valid_mask]] = dummy_preds[valid_mask]

    # 5. Fast O(1) Taxonomy Remap
    classes_4 = map_to_4_classes(point_labels)
    print(f"Mapped classes shape: {classes_4.shape}, unique: {np.unique(classes_4)}")
```

---

## 7. Architectural Guardrails & Enforced Rules

To maintain high throughput and avoid architectural drift, the following constraints are enforced across the codebase:

1. **Pure NumPy on Hot Path**:
   All operations between scan reading, range projection, unprojection, and grid insertion must execute strictly with NumPy. Do not use PyTorch tensors for grid operations or spatial sorting.
2. **Forbidden Dependencies**:
   - `open3d` is prohibited (unnecessary memory overhead and heavy C++ bindings).
   - `scikit-learn` RANSAC or DBSCAN is prohibited.
   - `filterpy` Kalman filters are prohibited.
3. **Decoupled Visualization**:
   The Streamlit dashboard (`src/fovmap/dashboard`) reads pre-saved `.npz` files from `outputs/`. It must never import `src/fovmap/pipeline` or execute model inference during user rendering loops.
4. **Exact Point Count Invariant**:
   Every point processed by the pipeline must either be assigned to a grid cell or accounted for in an explicit out-of-bounds counter (`assert counts.sum() == N`).
