# Foveated Semantic 2.5D LiDAR Mapping

> **Status: Work in Progress**  
> This project is actively being built for the Smart India Hackathon (SIH). The data ingestion, calibration, relative motion tracking, and class remapping modules are complete and tested. Full pipeline integration and range-image projection are currently underway.

Personal fork of our team's Smart India Hackathon (SIH) project. While this repository contains our full collaborative codebase, this fork showcases my work leading the **Data & Pose Subsystem (Person 1)**.

---

## What I Built (My Role: Person 1 — Data & Pose)

In our 6-person pipeline, my module is the entry point for the whole system: reading raw sensor files, lining up coordinate frames, calculating vehicle motion between scans, and simplifying semantic labels so the rest of the team can generate real-time 2.5D elevation maps.

### Key Contributions:

* **Fast Point Cloud Loader (`src/fovmap/data/loader.py`)**
  * Reads SemanticKITTI `.bin` files directly into NumPy arrays of shape `(N, 4)` (`[x, y, z, intensity]`).
  * Built strictly with NumPy to keep the ingestion path fast, avoiding the overhead of heavy 3D libraries like Open3D on the critical path.

* **Sensor Calibration & Coordinate Transforms**
  * Parses the Velodyne-to-camera extrinsic matrix `Tr` from `calib.txt`.
  * Converts 12-element lines from `poses.txt` into standard 4x4 homogeneous transformation matrices.

* **Relative Motion Tracking (`compute_T_rel`)**
  * KITTI ground-truth poses are recorded in the camera frame, but our pipeline operates in the LiDAR sensor's frame.
  * Solved the coordinate frame mismatch using matrix conjugation:
    `T_rel = inv(Tr) @ inv(pose_curr) @ pose_prev @ Tr`
  * Hands downstream mapping modules the exact relative movement between consecutive sweeps.

* **Sequence Iterator (`SemanticKITTILoader`)**
  * Clean, indexable dataset loader (`loader[i]`, `len(loader)`) that discovers files dynamically and handles frame-by-frame loading without hardcoded paths.

* **Instant 4-Class Taxonomy Remap (`src/fovmap/data/remap.py`)**
  * The segmentation model (SalsaNext) outputs 20 fine-grained classes, but our elevation grid only needs 4 navigation categories:
    `0: TERRAIN`, `1: DRIVABLE`, `2: STATIC`, `3: OBJECT`.
  * Implemented an O(1) vectorized lookup table (`_REMAP_LUT`) to map thousands of points instantly without slow loops.

* **Test & Verification Suite (`verify_gates.py`)**
  * Built a 7-stage verification script to test point dimensions, byte alignment, transform math, and label mapping before data gets handed to downstream teammates.

### Subsystem Progress:
- [x] Raw `.bin` point cloud ingestion (pure NumPy)
- [x] Calibration parsing & relative motion tracking (`T_rel`)
- [x] Dynamic multi-frame sequence loader
- [x] Fast 20-to-4 class taxonomy remapper
- [x] 7-stage verification test suite
- [ ] Spherical range-image projection `(64, 2048, C)` for neural network input (In progress)

---

## How the Pipeline Fits Together

```text
Raw LiDAR (.bin) + Poses & Calibration
         │
         ▼
[ My Code: Data & Pose ] ─────► Fast NumPy loading, calibration, & relative motion (T_rel)
         │
         ▼
[ Range Image Projection ] ───► Cylindrical projection into (64 x 2048) range image
         │
         ▼
[ SalsaNext Model ] ──────────► 20-class neural segmentation (team module)
         │
         ▼
[ Unprojection & Remapping ] ─► [ My Code ] Fast LUT remap: 20 classes -> 4 navigation classes
         │
         ▼
[ 2.5D Elevation Grid ] ──────► Concentric height rings & obstacle mapping (team module)
```

---

## Repository Structure

```text
.
├── README.md                           # This guide
├── README_FOVMAP_SALSANEXT.md          # Internal team guide & model details
├── verify_gates.py                     # [My Code] 7-stage test suite
├── foveated_lidar_architecture_v4.md   # Overall system architecture doc
├── person1_data_pose_implementation_plan.md  # My subsystem implementation notes
│
├── src/
│   └── fovmap/
│       └── data/
│           ├── loader.py               # [My Code] Binary scan parsing, poses, T_rel
│           └── remap.py                # [My Code] 20-to-4 class taxonomy lookup table
│
├── sample_kitti/                       # Sample dataset for quick testing
│   └── sequences/
│       └── 08/                         # 12 sample frames (velodyne, poses, calib)
│
├── SalsaNext-Fork/                     # Semantic segmentation network
└── docs/                               # System specs and references
```

---

## Running the Code

### 1. Setup

Clone the repository and jump in:

```bash
git clone https://github.com/CallidusK/Sihresources00.git
cd Sihresources00
```

Activate the virtual environment:

```bash
# Using the bundled venv
source .venv/bin/activate

# Or create a fresh one
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
```

### 2. Run the Verification Gates

Run the test suite against the included sample sequence (Sequence 08):

```bash
python verify_gates.py -s 08
```

You should see all 7 verification checks pass:

```text
=== GATE A1: Read one .bin scan (000000.bin) [Seq: 08] ===
 Gate A1 PASSED!
=== GATE A2: Load poses.txt [Seq: 08] ===
 Gate A2 PASSED!
=== GATE A3: Parse Tr from calib.txt [Seq: 08] ===
 Gate A3 PASSED!
=== GATE A4: SemanticKITTILoader Frame 0 Access [Seq: 08] ===
 Gate A4 PASSED!
=== GATE A5: Full-Sequence Multi-Frame Iteration [Seq: 08] ===
 Gate A5 PASSED! All frames iterated cleanly.
=== GATE B1: Relative Pose T_rel Verification [Seq: 08] ===
 Gate B1 PASSED! Relative poses match ground truth perfectly.
=== GATE REMAP: 19->4 Semantic Mapping ===
 Gate REMAP PASSED! Lookup table mapping is correct.
```

---

## What the Verification Gates Check

| Gate | Function / Component | What it verifies |
| :--- | :--- | :--- |
| **Gate A1** | `load_point_cloud` | Ensures points are `(N, 4)` `float32` and file size divides cleanly by 16 bytes. |
| **Gate A2** | `load_poses` | Checks that 3x4 pose rows convert into proper 4x4 transformation matrices. |
| **Gate A3** | `load_calib_tr` | Pulls and validates the 4x4 camera-to-LiDAR extrinsic matrix from `calib.txt`. |
| **Gate A4** | `SemanticKITTILoader[0]` | Verifies single-frame indexing and output shapes. |
| **Gate A5** | `SemanticKITTILoader` | Steps through the entire sequence without crashes or missing frames. |
| **Gate B1** | `compute_T_rel` | Tests relative pose computation between consecutive frames against ground truth. |
| **Gate REMAP** | `map_to_4_classes` | Confirms raw semantic predictions map correctly to the 4 navigation classes. |

---

## Team & Context

* **Project:** Foveated Semantic 2.5D LiDAR Mapping
* **Hackathon:** Smart India Hackathon (SIH)
* **Team Upstream Repo:** `yakshaarora2007-sketch/Sihresources00`
* **My Role:** Person 1 (Data & Pose Subsystem)
