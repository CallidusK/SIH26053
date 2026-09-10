"""SemanticKITTI Sequence 08 Data & Pose Loader.

Person 1 — Data & Pose pipeline.
"""

from pathlib import Path
import numpy as np


def load_point_cloud(
    bin_path: str | Path,
    dtype: type = np.float32,
    num_fields: int = 4,
) -> np.ndarray:
    """Loads a raw KITTI/SemanticKITTI LiDAR scan (.bin).

    The binary file is a flat stream of floats organized as
    repeating [x, y, z, intensity] tuples.

    Args:
        bin_path: Path to the .bin file.
        dtype: Data type of stored floats (default: np.float32).
        num_fields: Number of values per point (default: 4).

    Returns:
        np.ndarray of shape (N, num_fields) with the specified dtype.
    """
    bin_path = Path(bin_path)
    points = np.fromfile(bin_path, dtype=dtype)
    return points.reshape(-1, num_fields)


def load_poses(poses_path: str | Path) -> list[np.ndarray]:
    """Loads camera-frame poses from a KITTI poses.txt file.

    Each line in poses.txt contains 12 space-separated floats forming
    a 3x4 transformation matrix [R | t]. This function expands each entry
    into a homogeneous 4x4 transformation matrix by setting the bottom row
    to [0, 0, 0, 1].

    Args:
        poses_path: Path to poses.txt.

    Returns:
        List of 4x4 NumPy arrays with dtype float64.
    """
    poses_path = Path(poses_path)
    raw = np.loadtxt(poses_path, dtype=np.float64)
    raw_3x4 = raw.reshape(-1, 3, 4)

    poses: list[np.ndarray] = []
    for mat_3x4 in raw_3x4:
        mat_4x4 = np.eye(4, dtype=np.float64)
        mat_4x4[:3, :4] = mat_3x4
        poses.append(mat_4x4)

    return poses


def load_calib_tr(calib_path: str | Path) -> np.ndarray:
    """Parses Tr (Velodyne-to-camera extrinsic matrix) from a KITTI calib.txt file.

    Expands the 12 floats (3x4 [R | t]) into a 4x4 homogeneous transformation
    matrix with bottom row [0, 0, 0, 1].

    Args:
        calib_path: Path to calib.txt.

    Returns:
        np.ndarray of shape (4, 4) with dtype float64.
    """
    calib_path = Path(calib_path)
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith("Tr:"):
                values = [float(x) for x in line.strip().split()[1:]]
                if len(values) != 12:
                    raise ValueError(f"Expected 12 numbers for Tr, got {len(values)}")
                mat_4x4 = np.eye(4, dtype=np.float64)
                mat_4x4[:3, :4] = np.array(values, dtype=np.float64).reshape(3, 4)
                return mat_4x4

    raise ValueError(f"'Tr:' line not found in {calib_path}")


def compute_T_rel(
    pose_curr: np.ndarray,
    pose_prev: np.ndarray,
    Tr: np.ndarray,
) -> np.ndarray:
    """Computes relative LiDAR-frame transformation from previous frame to current frame.

    Because KITTI poses are in the camera coordinate frame, the LiDAR-frame relative
    motion requires the extrinsic conjugation:
        T_rel = inv(Tr) @ inv(pose_curr) @ pose_prev @ Tr

    Args:
        pose_curr: 4x4 camera-frame pose at current frame (N).
        pose_prev: 4x4 camera-frame pose at previous frame (N-1).
        Tr: 4x4 Velodyne-to-camera extrinsic transformation matrix.

    Returns:
        np.ndarray of shape (4, 4) with dtype float64.
    """
    if np.allclose(pose_curr, pose_prev, atol=1e-6):
        return np.eye(4, dtype=np.float64)

    inv_Tr = np.linalg.inv(Tr)
    inv_pose_curr = np.linalg.inv(pose_curr)
    return inv_Tr @ inv_pose_curr @ pose_prev @ Tr


class SemanticKITTILoader:
    """Multi-frame dataset loader for SemanticKITTI sequences.

    Given a sequence directory (e.g. sequences/08), this loader:
      1. Automatically discovers all scan (.bin) files without hardcoded file naming.
      2. Loads poses from poses.txt.
      3. Loads extrinsic calibration Tr from calib.txt.
      4. Supports len() and indexing to iterate through the entire sequence.
    """

    def __init__(self, sequence_dir: str | Path):
        self.sequence_dir = Path(sequence_dir)
        self.velodyne_dir = self.sequence_dir / "velodyne"
        self.poses_file = self.sequence_dir / "poses.txt"
        self.calib_file = self.sequence_dir / "calib.txt"

        if not self.velodyne_dir.is_dir():
            raise FileNotFoundError(f"Velodyne directory not found: {self.velodyne_dir}")

        self.scan_files: list[Path] = sorted(self.velodyne_dir.glob("*.bin"))
        if not self.scan_files:
            raise FileNotFoundError(f"No .bin files found in {self.velodyne_dir}")

        if not self.poses_file.exists():
            raise FileNotFoundError(f"Poses file not found: {self.poses_file}")
        self.poses = load_poses(self.poses_file)

        if len(self.poses) != len(self.scan_files):
            raise ValueError(
                f"Mismatch: {len(self.scan_files)} .bin files found but {len(self.poses)} poses parsed."
            )

        if not self.calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {self.calib_file}")
        self.Tr = load_calib_tr(self.calib_file)

    def __len__(self) -> int:
        """Total number of frames in the sequence."""
        return len(self.scan_files)

    def __getitem__(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Loads and returns all necessary data for frame index `idx`.

        Args:
            idx: Frame index in range [0, len(loader) - 1].

        Returns:
            Tuple of:
                - points: np.ndarray of shape (N, 4), dtype float32 [x, y, z, intensity]
                - pose: np.ndarray of shape (4, 4), dtype float64 (current camera-frame pose)
                - pose_prev: np.ndarray of shape (4, 4), dtype float64 (previous frame pose, or identity for idx=0)
                - Tr: np.ndarray of shape (4, 4), dtype float64 (velodyne-to-camera extrinsic)
                - frame_id: int (index of the frame)
        """
        if idx < 0 or idx >= len(self.scan_files):
            raise IndexError(f"Frame index {idx} out of range [0, {len(self.scan_files) - 1}]")

        points = load_point_cloud(self.scan_files[idx])
        pose = self.poses[idx]
        pose_prev = self.poses[idx - 1] if idx > 0 else np.eye(4, dtype=np.float64)

        return points, pose, pose_prev, self.Tr, idx

    def get_T_rel(self, idx: int) -> np.ndarray:
        """Computes relative LiDAR-frame pose for frame index `idx`."""
        if idx == 0:
            return np.eye(4, dtype=np.float64)
        _, pose_curr, pose_prev, Tr, _ = self[idx]
        return compute_T_rel(pose_curr, pose_prev, Tr)