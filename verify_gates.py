"""Verification script for Person 1 Data & Pose pipeline gates."""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to the Python path so we can import fovmap
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fovmap.data.loader import (
    load_point_cloud,
    load_poses,
    load_calib_tr,
    compute_T_rel,
    SemanticKITTILoader,
)
from fovmap.data.remap import map_to_4_classes, LEARNING_MAP_INV


def check_gate_a1(
    data_root: Path,
    seq_id: str = "04",
    dtype: type = np.float32,
    num_fields: int = 4,
):
    print(f"=== GATE A1: Read one .bin scan (000000.bin) [Seq: {seq_id}] ===")

    sample_bin = data_root / "sequences" / seq_id / "velodyne" / "000000.bin"
    if not sample_bin.exists():
        print(f"Error: File not found: {sample_bin}")
        print(f"Tip: If running against minimal sample data, pass --seq 08")
        sys.exit(1)

    points = load_point_cloud(sample_bin, dtype=dtype, num_fields=num_fields)

    print(f"points.shape : {points.shape}")
    print(f"points.dtype : {points.dtype}")
    print(f"x range      : [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] m")
    print(f"y range      : [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] m")
    print(f"z range      : [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m")
    print(f"intensity    : [{points[:, 3].min():.3f}, {points[:, 3].max():.3f}]")

    # Dynamically verify file size using the dtype's item size in bytes
    bytes_per_field = np.dtype(dtype).itemsize
    bytes_per_point = num_fields * bytes_per_field
    file_size = sample_bin.stat().st_size
    assert file_size % bytes_per_point == 0, (
        f"File size {file_size} is not divisible by {bytes_per_point} bytes!"
    )
    expected_points = file_size // bytes_per_point

    # Gate checks (dynamic for any valid scan and data type)
    assert points.ndim == 2 and points.shape[1] == num_fields, (
        f"Expected shape (N, {num_fields}), got {points.shape}"
    )
    assert points.shape[0] == expected_points, (
        f"Expected {expected_points} points from file size, got {points.shape[0]}"
    )
    assert points.dtype == dtype, f"Expected {dtype}, got {points.dtype}"
    assert np.all(np.isfinite(points)), "Point cloud contains NaN or Inf values!"
    assert 0.0 <= points[:, 3].min() and points[:, 3].max() <= 1.0, (
        "intensity out of expected bounds [0.0, 1.0]"
    )

    print(" Gate A1 PASSED!")


def check_gate_a2(data_root: Path, seq_id: str = "04"):
    print(f"\n=== GATE A2: Load poses.txt [Seq: {seq_id}] ===")

    sequence_dir = data_root / "sequences" / seq_id
    poses_file = sequence_dir / "poses.txt"
    velodyne_dir = sequence_dir / "velodyne"

    if not poses_file.exists():
        print(f"Error: File not found: {poses_file}")
        print(f"Tip: If running against minimal sample data, pass --seq 08")
        sys.exit(1)

    raw = np.loadtxt(poses_file, dtype=np.float64)
    print(f"raw poses.shape : {raw.shape}")
    print(f"raw poses.dtype : {raw.dtype}")

    poses = load_poses(poses_file)
    print(f"num poses       : {len(poses)}")
    print(f"pose[0].shape   : {poses[0].shape}")
    print(f"pose[0].dtype   : {poses[0].dtype}")

    bin_files = sorted(velodyne_dir.glob("*.bin"))
    print(f"num .bin files  : {len(bin_files)}")

    # 1. Shape of raw poses
    assert raw.ndim == 2 and raw.shape[1] == 12, (
        f"Expected shape (N, 12), got {raw.shape}"
    )
    assert raw.shape[0] == len(bin_files), (
        f"Expected {len(bin_files)} lines, got {raw.shape[0]}"
    )

    # 2. Number of parsed poses equals number of .bin files
    assert len(poses) == len(bin_files), (
        f"Expected {len(bin_files)} poses, got {len(poses)}"
    )

    # 3. pose[0] is identity
    assert poses[0].shape == (4, 4), f"Expected (4, 4), got {poses[0].shape}"
    assert np.allclose(poses[0], np.eye(4, dtype=np.float64), atol=1e-6), (
        "pose[0] is not identity!"
    )

    # 4. Consecutive translations differ by a sane amount (~1-2 m)
    translations = [p[:3, 3] for p in poses]
    diffs = [
        np.linalg.norm(translations[i] - translations[i - 1])
        for i in range(1, len(translations))
    ]
    print(
        f"consecutive translation distances (m): [{min(diffs):.3f}, {max(diffs):.3f}]"
    )
    for idx, d in enumerate(diffs, start=1):
        assert 0.1 <= d <= 5.0, (
            f"Translation step between pose {idx-1} and {idx} is unexpected: {d:.3f} m"
        )

    print(" Gate A2 PASSED!")


def check_gate_a3(data_root: Path, seq_id: str = "04"):
    print(f"\n=== GATE A3: Parse Tr from calib.txt [Seq: {seq_id}] ===")
    calib_file = data_root / "sequences" / seq_id / "calib.txt"
    if not calib_file.exists():
        print(f"Error: File not found: {calib_file}")
        print(f"Tip: If running against minimal sample data, pass --seq 08")
        sys.exit(1)

    Tr = load_calib_tr(calib_file)
    print(f"Tr.shape : {Tr.shape}")
    print(f"Tr.dtype : {Tr.dtype}")
    print(f"Tr matrix:\n{Tr}")

    assert Tr.shape == (4, 4), f"Expected Tr to be (4, 4), got {Tr.shape}"
    assert Tr.dtype == np.float64, f"Expected Tr dtype float64, got {Tr.dtype}"
    assert np.allclose(Tr[3, :], [0, 0, 0, 1]), (
        f"Expected bottom row [0, 0, 0, 1], got {Tr[3, :]}"
    )

    R = Tr[:3, :3]
    det_R = float(np.linalg.det(R))
    print(f"det(R_velo_to_cam) : {det_R:.6f}")
    assert np.allclose(det_R, 1.0, atol=1e-3), (
        f"Rotation determinant not ~1.0: {det_R}"
    )

    print(" Gate A3 PASSED!")


def check_gate_a4(data_root: Path, seq_id: str = "04"):
    print(f"\n=== GATE A4: SemanticKITTILoader Frame 0 Access [Seq: {seq_id}] ===")
    seq_dir = data_root / "sequences" / seq_id
    if not seq_dir.exists():
        print(f"Error: Sequence directory not found: {seq_dir}")
        print(f"Tip: If running against minimal sample data, pass --seq 08")
        sys.exit(1)

    loader = SemanticKITTILoader(seq_dir)

    print(f"len(loader) : {len(loader)} frames")
    assert len(loader) > 0, "Loader should have at least 1 frame"

    points, pose, pose_prev, Tr, frame_id = loader[0]
    print(f"loader[0] frame_id  : {frame_id}")
    print(f"loader[0] points    : shape={points.shape}, dtype={points.dtype}")
    print(f"loader[0] pose      : shape={pose.shape}, dtype={pose.dtype}")
    print(f"loader[0] pose_prev : shape={pose_prev.shape}, dtype={pose_prev.dtype}")
    print(f"loader[0] Tr        : shape={Tr.shape}, dtype={Tr.dtype}")

    assert frame_id == 0
    assert points.shape[0] > 0, "Frame 0 has no points"
    assert points.shape[1] == 4
    assert points.dtype == np.float32
    assert pose.shape == (4, 4)
    assert pose_prev.shape == (4, 4)
    assert np.allclose(pose_prev, np.eye(4, dtype=np.float64)), (
        "pose_prev for frame 0 should be identity"
    )
    assert Tr.shape == (4, 4)

    print(" Gate A4 PASSED!")


def check_gate_a5(data_root: Path, seq_id: str = "04"):
    print(f"\n=== GATE A5: Full-Sequence Multi-Frame Iteration [Seq: {seq_id}] ===")
    seq_dir = data_root / "sequences" / seq_id
    if not seq_dir.exists():
        print(f"Error: Sequence directory not found: {seq_dir}")
        print(f"Tip: If running against minimal sample data, pass --seq 08")
        sys.exit(1)

    loader = SemanticKITTILoader(seq_dir)

    print(f"Iterating through all {len(loader)} frames...")
    for idx, (points, pose, pose_prev, Tr, frame_id) in enumerate(loader):
        assert frame_id == idx
        assert points.shape[0] > 0, f"Frame {idx}: got empty point cloud"
        assert points.shape[1] == 4
        assert points.dtype == np.float32
        assert pose.shape == (4, 4)
        print(
            f"  Frame {idx:02d}: {points.shape[0]} points, pose translation = "
            f"[{pose[0,3]:.2f}, {pose[1,3]:.2f}, {pose[2,3]:.2f}] m"
        )

    print(" Gate A5 PASSED! All frames iterated cleanly.")


def check_gate_b1(data_root: Path, seq_id: str = "04", gt_file: Path | None = None):
    print(f"\n=== GATE B1: Relative Pose T_rel Verification [Seq: {seq_id}] ===")
    seq_dir = data_root / "sequences" / seq_id
    if not seq_dir.exists():
        print(f"Error: Sequence directory not found: {seq_dir}")
        print(f"Tip: If running against minimal sample data, pass --seq 08")
        sys.exit(1)

    loader = SemanticKITTILoader(seq_dir)

    # Dynamically locate ground truth file matching the current sequence
    resolved_gt: Path | None = None
    if gt_file is not None:
        if not gt_file.exists():
            print(f"Error: Explicitly specified ground truth file not found: {gt_file}")
            sys.exit(1)
        resolved_gt = gt_file
    else:
        candidate_paths = [
            seq_dir / f"GROUND_TRUTH_T_rel_{seq_id}.txt",
            seq_dir / "GROUND_TRUTH_T_rel.txt",
            data_root / f"GROUND_TRUTH_T_rel_{seq_id}.txt",
            data_root / f"GROUND_TRUTH_T_rel_seq{seq_id}.txt",
        ]

        for p in candidate_paths:
            if p.exists():
                resolved_gt = p
                break

    if resolved_gt is None:
        print(
            f"Error: Ground truth relative pose file not found for sequence {seq_id}."
        )
        print(
            "Tip: Supply an explicit ground truth file via --gt-file, "
            "or ensure GROUND_TRUTH_T_rel.txt exists in the sequence directory."
        )
        sys.exit(1)

    # Parse ground truth T_rel matrices
    gt_matrices: dict[int, np.ndarray] = {}
    with open(resolved_gt, "r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("frame ") and "->" in line:
            # e.g. "frame 0->1:" -> frame 1
            target_frame = int(line.split("->")[1].replace(":", "").strip())
            mat_rows = []
            for r in range(1, 5):
                mat_rows.append([float(x) for x in lines[i + r].split()])
            gt_matrices[target_frame] = np.array(mat_rows, dtype=np.float64)
            i += 5
        else:
            i += 1

    if not gt_matrices:
        print(
            f"Error: No valid relative pose transitions found in {resolved_gt}."
        )
        sys.exit(1)

    # Check frame 0 is identity
    t_rel_0 = loader.get_T_rel(0)
    assert np.allclose(t_rel_0, np.eye(4, dtype=np.float64)), (
        "Frame 0 T_rel should be identity"
    )

    # Check frames 1..11 against ground truth
    print(
        f"Comparing computed T_rel against ground truth for {len(gt_matrices)} transitions..."
    )
    for frame_idx, gt_T in sorted(gt_matrices.items()):
        computed_T = loader.get_T_rel(frame_idx)
        assert np.allclose(computed_T, gt_T, atol=1e-4), (
            f"T_rel mismatch at frame {frame_idx}!\nComputed:\n{computed_T}\nExpected:\n{gt_T}"
        )
        print(
            f"  Frame {frame_idx - 1}->{frame_idx}: MATCH (max abs diff = {np.max(np.abs(computed_T - gt_T)):.2e})"
        )

    print(" Gate B1 PASSED! Relative poses match ground truth perfectly.")


def check_gate_remap():
    print("\n=== GATE REMAP: 19->4 Semantic Mapping ===")

    # Test all 260 possible classes
    test_classes = np.arange(260, dtype=np.uint8)
    mapped = map_to_4_classes(test_classes)

    # Verify some known cases based on the plan
    assert mapped[40] == 1, "Road (40) should be DRIVABLE (1)"
    assert mapped[44] == 1, "Parking (44) should be DRIVABLE (1)"
    assert mapped[60] == 1, "Lane marking (60) should be DRIVABLE (1)"

    assert mapped[50] == 2, "Building (50) should be STATIC (2)"
    assert mapped[80] == 2, "Pole (80) should be STATIC (2)"

    assert mapped[10] == 3, "Car (10) should be OBJECT (3)"
    assert mapped[30] == 3, "Person (30) should be OBJECT (3)"
    assert mapped[252] == 3, "Moving car (252) should be OBJECT (3)"

    assert mapped[0] == 0, "Unlabeled (0) should be TERRAIN (0)"
    assert mapped[72] == 0, "Terrain (72) should be TERRAIN (0)"
    assert mapped[48] == 0, "Sidewalk (48) should be TERRAIN (0)"

    # Verify inverse learning map lookup table (20 classes)
    assert len(LEARNING_MAP_INV) == 20, "LEARNING_MAP_INV must have 20 entries"
    assert LEARNING_MAP_INV[1] == 10, "Learning class 1 (car) should map to raw ID 10"
    assert LEARNING_MAP_INV[9] == 40, "Learning class 9 (road) should map to raw ID 40"
    assert LEARNING_MAP_INV[13] == 50, "Learning class 13 (building) should map to raw ID 50"
    assert LEARNING_MAP_INV[17] == 72, "Learning class 17 (terrain) should map to raw ID 72"

    print(" Gate REMAP PASSED! Lookup table mapping is correct.")


def run_all_gates(
    data_root: Path,
    seq_id: str = "04",
    gt_file: Path | None = None,
):
    """Execute all Person 1 verification gates for a given sequence."""
    check_gate_a1(data_root, seq_id=seq_id)
    check_gate_a2(data_root, seq_id=seq_id)
    check_gate_a3(data_root, seq_id=seq_id)
    check_gate_a4(data_root, seq_id=seq_id)
    check_gate_a5(data_root, seq_id=seq_id)
    check_gate_b1(data_root, seq_id=seq_id, gt_file=gt_file)
    check_gate_remap()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verification script for Person 1 Data & Pose pipeline gates."
    )
    parser.add_argument(
        "--data-root",
        "-d",
        type=Path,
        default=Path(__file__).resolve().parent / "sample_kitti",
        help="Path to SemanticKITTI root containing sequences/ directory (default: sample_kitti)",
    )
    parser.add_argument(
        "--seq",
        "-s",
        type=str,
        default="04",
        help="Sequence ID to test (default: '04')",
    )
    parser.add_argument(
        "--gt-file",
        "-g",
        type=Path,
        default=None,
        help="Optional explicit path to ground-truth T_rel file (default: auto-detect by seq_id)",
    )

    args = parser.parse_args()
    run_all_gates(
        data_root=args.data_root,
        seq_id=args.seq,
        gt_file=args.gt_file,
    )
