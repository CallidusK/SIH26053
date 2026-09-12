"""
Mock grid generator for the LiDAR dashboard.

Produces data shaped exactly like what Dept 2 (grid engine) + Dept 3
(terrain/dynamics/fusion) will eventually hand off:

    grid: dict[(ring_id, row, col) -> Cell]

Swap this module out for the real pipeline later — app.py only calls
`mock_grid()` / `mock_uniform_grid()`, so as long as the real code returns
the same dict-of-Cell shape, nothing in app.py needs to change.
"""

import random
from dataclasses import dataclass


# 12-class taxonomy referenced in Dept 1 (PointNet++ segmentation)
SEMANTIC_CLASS_NAMES = [
    "road",
    "sidewalk",
    "curb",
    "wall",
    "pole",
    "vehicle",
    "pedestrian",
    "cyclist",
    "vegetation",
    "building",
    "terrain",
    "unknown",
]

TRAVERSABILITY_LEVELS = ["safe", "caution", "non_drivable"]

# Ring 0 = closest/finest band, ring 3 = farthest/coarsest band (see Dept 2)
RES_TABLE = [0.05, 0.10, 0.25, 0.50]  # meters per cell, per ring
RING_BOUNDS = [(0.0, 10.0), (10.0, 25.0), (25.0, 50.0), (50.0, 100.0)]


@dataclass
class Cell:
    elevation_mean: float
    elevation_min: float
    elevation_max: float
    height_var: float
    semantic_class: int
    semantic_conf: float
    occupancy_prob: float
    dynamic_prob: float
    traversability: str
    point_count: int
    velocity: tuple = None


def _random_cell(rng: random.Random, ring_id: int) -> Cell:
    elevation_mean = rng.uniform(-0.5, 2.5)
    height_var = rng.uniform(0.0, 0.05)
    semantic_class = rng.randint(0, len(SEMANTIC_CLASS_NAMES) - 1)
    dynamic_prob = rng.random() if semantic_class in (5, 6, 7) else rng.uniform(0, 0.1)

    slope_like = height_var * 10 + rng.uniform(0, 0.15)
    if slope_like < 0.08:
        trav = "safe"
    elif slope_like < 0.2:
        trav = "caution"
    else:
        trav = "non_drivable"

    return Cell(
        elevation_mean=elevation_mean,
        elevation_min=elevation_mean - rng.uniform(0, 0.3),
        elevation_max=elevation_mean + rng.uniform(0, 0.3),
        height_var=height_var,
        semantic_class=semantic_class,
        semantic_conf=rng.uniform(0.6, 0.99),
        occupancy_prob=rng.uniform(0.4, 0.99),
        dynamic_prob=dynamic_prob,
        traversability=trav,
        point_count=rng.randint(1, 60),
        velocity=(rng.uniform(-2, 2), rng.uniform(-2, 2)) if dynamic_prob > 0.5 else None,
    )


def mock_grid(n_cells: int = 1500, grid_radius: int = 100, seed: int = 0):
    """
    Adaptive variable-resolution mock grid: dict[(ring, row, col) -> Cell].
    Row and col are indices in that ring's metric grid. Each ring is sampled
    from its radial band so the mock preserves the physical foveation layout.
    """
    rng = random.Random(seed)
    grid = {}
    ring_weights = [0.45, 0.3, 0.17, 0.08]  # more cells near the sensor

    attempts = 0
    while len(grid) < n_cells and attempts < n_cells * 5:
        attempts += 1
        ring_id = rng.choices(range(4), weights=ring_weights, k=1)[0]
        resolution = RES_TABLE[ring_id]
        inner_radius, outer_radius = RING_BOUNDS[ring_id]
        outer_radius = min(outer_radius, grid_radius)
        if outer_radius <= inner_radius:
            continue

        max_index = int(outer_radius / resolution)
        row = rng.randint(-max_index, max_index)
        col = rng.randint(-max_index, max_index)
        x = col * resolution
        y = row * resolution
        radius = (x * x + y * y) ** 0.5
        if not inner_radius <= radius < outer_radius:
            continue
        key = (ring_id, row, col)
        if key not in grid:
            grid[key] = _random_cell(rng, ring_id)

    return grid


def mock_uniform_grid(res_label: str, n_cells: int, grid_radius: int = 60, seed: int = 0):
    """
    A uniform-resolution baseline grid (all cells 'ring 0') for the
    Uniform-vs-Adaptive comparison view. res_label is cosmetic only here.
    """
    rng = random.Random(seed + hash(res_label) % 1000)
    grid = {}
    attempts = 0
    while len(grid) < n_cells and attempts < n_cells * 5:
        attempts += 1
        row = rng.randint(-grid_radius, grid_radius)
        col = rng.randint(-grid_radius, grid_radius)
        key = (0, row, col)
        if key not in grid:
            grid[key] = _random_cell(rng, 0)
    return grid
