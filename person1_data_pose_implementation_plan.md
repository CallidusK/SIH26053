# Person 1 --- Data & Pose · Implementation Plan

> Brief for an AI coding agent (Antigravity IDE). Build this
> **incrementally, one task at a time, stopping at each gate for the
> human to verify** before continuing. Prefer small, readable code. **Do
> not dump the whole loader in one shot --- the human is learning this
> and needs to check each piece.**

------------------------------------------------------------------------

## 1. Project context

Part of a 6-person hackathon project: **Foveated Semantic 2.5D LiDAR
Mapping** --- a dataset-*replay* pipeline (no live sensor) over
**SemanticKITTI sequence 08** (4,007 scans). It runs as one per-scan
loop; each person owns some steps.

**I am Person 1 --- Data & Pose.** I am the front door of the pipeline:
I read raw files and produce clean data + relative motion for everyone
else. I do **not** do the neural net, the grid, the fusion, or the
dashboard.

My steps in the pipeline: **Step 1 (read)**, **Step 2 input side
(range-image projection)**, **Step 5 (compute T_rel)**. Steps 2--4's
labelling and everything from Step 6 on are other people's work.

------------------------------------------------------------------------

## 2. My deliverable

A dataset loader that, given a frame index `N`, returns clean in-memory
data with **no manual file naming**, and can iterate all frames end to
end:

-   `points` --- NumPy array, shape `(num_points, 4)`, `float32` =
    `[x, y, z, intensity]`
-   `T_rel` --- `(4, 4)` `float64` relative-pose matrix (Phase B)
-   `frame_id` --- int
-   (Phase C) a `(64, 2048, C)` range image for the segmentation model

**Hot-path rule:** NumPy only. No Open3D in the timed path.

------------------------------------------------------------------------

## 3. Locked-in facts (do not re-derive)

-   **`.bin` format:** flat `float32` stream, `[x,y,z,intensity]`
    repeated → `np.fromfile(path, dtype=np.float32).reshape(-1, 4)`.
-   **`poses.txt`:** each line = 12 numbers = a 3×4 `[R|t]`; make 4×4 by
    placing it in the top 3 rows of `np.eye(4)`.
-   **`calib.txt` `Tr:` line:** the velodyne→cam0 extrinsic (3×4 → 4×4
    the same way).
-   **Frame convention:** KITTI poses are in the **camera frame**,
    confirmed for this data. So the LiDAR-frame relative pose needs the
    extrinsic conjugation ("sandwich").
-   **T_rel formula (camera-frame case):**
    `T_rel = inv(Tr) @ inv(pose_N) @ pose_{N-1} @ Tr` (If poses were
    LiDAR-frame it would collapse to `inv(pose_N) @ pose_{N-1}`.)
-   **Frame 0:** no previous scan → `T_rel = np.eye(4)`.
-   **What T_rel is for (context only --- NOT my code):** the current
    scan gets **no transform**; T_rel is consumed downstream (Step 8,
    Person 5) to re-align the *stored map* into the current frame. My
    job ends at producing a correct T_rel.
-   **Float comparisons:** always `np.allclose(a, b, atol=1e-6)`, never
    `==`.

------------------------------------------------------------------------

## 4. Task breakdown (each task has a gate that must pass)

### Phase A --- Step 1: reading

**A1. Read one `.bin` scan.** Gate on sample `000000.bin`:
`points.shape == (17983, 4)`, `points.dtype == float32`, x/y within
\~±50 m, z roughly −4.6...6.2, intensity within 0.05...0.6.

**A2. Load `poses.txt` into a list of 4×4 matrices.** Gate:
`np.loadtxt('poses.txt').shape == (12, 12)`; number of poses **equals**
number of `.bin` files; `pose[0]` is identity (`[R|t]` = I, t = 0);
consecutive translations differ by a sane amount (\~1--2 m), not tens of
metres.

**A3. Parse `Tr` from `calib.txt` into a 4×4.** Gate:
`np.linalg.det(Tr[:3,:3])` ≈ 1.0.

**A4. Wrap A1--A3 into a loader class.** `__init__` loads poses + `Tr`
once; `__len__` returns frame count; `__getitem__(idx)` returns
`points[idx]`, `pose[idx]`, `pose[idx-1]`, `Tr`, `frame_id`. Decide what
`pose[idx-1]` means at `idx == 0` (identity is the standard choice). No
hardcoded absolute paths. Gate: `loader[0]` returns the right shapes
with no manual file naming.

**A5. Full-sequence iteration.** Gate:
`for i in range(len(loader)): loader[i]` runs through all frames with
zero crashes and zero manual intervention; per-frame point counts print
sanely (17983, 19352, 19803, 17456, 18667, 18894, 17978, 17689, 19117,
18368, 16808, 18208).

### Phase B --- Step 5: compute T_rel

**B1.** Add a method that returns the `(4,4)` T_rel for frame `N` using
the camera-frame formula above; `np.eye(4)` for frame 0. Gate: compare
each computed T_rel against `GROUND_TRUTH_T_rel.txt` (shipped with the
fixture) using `np.allclose`. They must match. A synthetic hand-checked
point transformed by T_rel should also land where expected.

### Phase C --- Step 2 input side: range-image projection

**C1.** Project the `(N,4)` cloud onto a `64 × 2048` range image: row =
laser ring, column = azimuth (`atan2(y, x)`); store range (+ x, y, z,
intensity as channels). This is the input format the segmentation model
(Persons 2/3) needs. **Open coordination question to resolve first:**
hand Persons 2/3 the raw `(N,4)` points and let the SalsaNext repo do
its own projection, **or** own the projection here and hand them a ready
range image. Pick one before building C1.

------------------------------------------------------------------------

## 5. Test fixture (build/verify against this first)

A miniature KITTI in the real layout ships as `sample_kitti.zip`. Point
the loader at `sample_kitti/sequences/08`:

    sequences/08/
      velodyne/000000.bin … 000011.bin   float32 [x,y,z,intensity]
      labels/000000.label … 000011.label uint32 raw SemanticKITTI IDs (10,40,48,50,70,72)
      poses.txt    12 lines, camera-frame, pose_0 = identity
      calib.txt    real Tr line
      times.txt
    GROUND_TRUTH_T_rel.txt                answer key for Phase B — do not read into the loader

12 frames, varying point counts by design (so shape checks are
meaningful). Everything above in section 4's gates was measured on this
fixture.

------------------------------------------------------------------------

## 6. Working constraints for the agent

-   Build one task at a time; **stop at each gate** and report the
    printed values.
-   Print `.shape` and `.dtype` on everything loaded --- catches most
    bugs.
-   Start on **one** `.bin` before looping over the sequence.
-   Use `np.allclose`, never `==`, for float checks.
-   No hardcoded absolute paths (`C:\Users\...`); take the sequence path
    as an argument.
-   Keep the returned `T_rel` a `(4,4)` NumPy array, never a list.
-   Explain each piece briefly as you go; the human is building
    understanding, not just code.

------------------------------------------------------------------------

## 7. Out of scope (do not build)

-   SalsaNext training/inference, torch patching --- Persons 2/3.
-   Median filter / gather / **applying** the 19→4 remap --- Persons
    2/3. (I *do* own writing + unit-testing the 19→4 remap **table**
    against the official `semantic-kitti-api` config, but that is a
    separate small task, not the loader.)
-   Foveated grid engine --- Person 4. Rating / fusion / re-binning with
    T_rel --- Person 5.
-   Dashboard --- Person 6.
