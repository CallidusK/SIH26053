# Foveated Grid Engine: Handover to Person 5

This document explains the `grid_engine.py` module (Step 6) and how it provides the foundation for Person 5's responsibilities: Grid Rating, Fusion, and Dynamics (Steps 7, 8, and 9).

As Person 4, I've built the core novelty of this project—the foveated radial grid—ensuring zero data loss and exact nested-lattice quantization. Here is how you (Person 5) can utilize it to build out terrain scoring, robot-centric fusion, and dynamic object detection.

## 1. Where this file fits in the Architecture

Currently, `grid_engine.py` is at the root. For a clean architecture, it should be moved into the main package directory, for example: `src/fovmap/core/grid_engine.py` or `src/fovmap/grid_engine.py`. This ensures it can be imported cleanly across the pipeline without polluting the root directory.

## 2. The Core Data Structure: `CELL_DTYPE`

The entire grid operates on a structured NumPy array, `CELL_DTYPE`. This is the schema for the "map" you will be modifying and updating over time.

```python
CELL_DTYPE = np.dtype([
    ('cell_key', np.int64),       # Unique packed integer: (ring_id | row | col)
    ('ring_id', np.uint8),        # 0=5cm, 1=10cm, 2=25cm, 3=50cm
    ('row', np.int32),            # Local grid Y index
    ('col', np.int32),            # Local grid X index
    ('point_count', np.uint32),   # Points assigned to this cell
    ('z_mean', np.float32),       # Mean elevation
    ('z_min', np.float32),        # Minimum elevation
    ('z_max', np.float32),        # Maximum elevation
    ('semantic_label', np.uint8), # 0=DRIVABLE, 1=TERRAIN, 2=STATIC, 3=OBJECT
    ('confidence', np.float32),   # Segmentation confidence
    ('dynamic_flag', np.bool_),   # YOU WILL SET THIS (Step 9)
    ('timestamp', np.uint32),     # Scan timestamp for ghost decay
])
```

## 3. How to use it for your tasks

### Step 7: Cell Rating (Elevation + Traversability)
The function `build_cell_schema(...)` takes raw points, semantics, and confidences and aggregates them into unique cells, returning a structured array of `CELL_DTYPE`. 

- **Your Task:** You need to compute obstacle height and traversability.
- **How to do it:** The grid already gives you `z_mean`, `z_min`, and `z_max`. You can iterate or vectorize over the generated cells to find local ground height (using cells where `semantic_label == 0` i.e. DRIVABLE). Then, calculate obstacle height as `cell['z_max'] - local_ground`.

### Step 8: Robot-Centric Fusion (Re-binning)
I have provided `rebin_stored_map(stored_cells, T_rel)` specifically to handle the structural part of your fusion step.
- **What it does:** It takes the map from the *previous* frame (`stored_cells`) and the relative motion (`T_rel`), transforms the cells into the current robot-centric frame, and exactly re-quantizes them into the new rings.
- **Why this helps you:** Because the grid resolutions are exact integer multiples (5cm -> 10cm -> 25cm -> 50cm) and the boundaries are strictly half-open `[low, high)`, coarsening and refining are exact sum/union operations without sub-pixel blur.
- **Your Task:** You will use this re-binned map as the baseline to compare against the *current* frame's cells.

### Step 9: Change Detection & Log-Odds Fusion
Once you have the `current_cells` (from `build_cell_schema`) and the `rebinned_map` (from `rebin_stored_map`) in the same coordinate frame:
- **Change Detection (Dynamics):** Compare `current_cells` and `rebinned_map`. If a cell flipped from road to object (or vice versa), and meets your confidence/evidence thresholds, you set `cell['dynamic_flag'] = True`.
- **Log-Odds Fusion:** After excluding dynamic cells, you will accumulate the static observations using standard log-odds math, decay them over time, and prune cells that are >100m away.

## 4. Benchmarking Helper
I have also provided `compute_memory_table(...)` which runs the schema builder in three modes: Foveated, Uniform 5cm, and Uniform 50cm. This gives you the cell counts and memory footprint (KB) out of the box, fulfilling the central evidence requirement for your benchmarking harness.

## Summary Checklist for Person 5
- [ ] Call `build_cell_schema` to get current frame cells.
- [ ] Write logic to compute local ground using `z_max` and `z_min` for Step 7.
- [ ] Call `rebin_stored_map` to shift last frame's map into the current frame.
- [ ] Compare current frame cells against the rebinned map to flag `dynamic_flag = True`.
- [ ] Implement log-odds accumulation for the persistent static map.
- [ ] Wire the memory and latency outputs into your benchmarking suite.
