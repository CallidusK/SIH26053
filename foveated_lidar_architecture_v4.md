# Foveated Semantic 2.5D LiDAR Mapping — Repository Architecture (v4)

> Canonical build: **SalsaNext** (RangeFormer fallback) · NumPy-only hot path · Robot-centric log-odds fusion (Fankhauser et al., ICRA 2014) · No Open3D / RANSAC / Kalman / world-frame accumulation — **enforced by CI, not just documented.**

This document is the single source of truth for repo layout, branching, the 7-day gate timeline, and day-to-day usage. Treat it like the plan itself: **tracked hand-edits only, no regeneration wholesale.**

---

## What changed in this revision (v3 → v4)

- **`contracts.py` gets a change policy**, not just a location. A `SCHEMA_VERSION` constant plus an additive-vs-breaking rule means the "all-six review" requirement only fires when it actually needs to — additive fields ship on a single owner's PR as long as the backward-compatibility test passes.
- **`tests/architecture/` becomes a real DAG check**, not three spot-checks. Every package's allowed imports are enumerated once and enforced by static analysis, so "viz never touches the timed path" is one instance of a general rule, not a special case.
- **`docs/resources.md` and `docs/risk_notes.md` added.** The plan's Consolidated Resource Table and Risk & Load Notes sections previously lived only as comments scattered through code — now they have a tracked, single home.
- **`GATES.md`** — an append-only, git-tracked ledger that `make gate-dayN` writes to on every pass. Day-7 traceability becomes "read this file," not "reconstruct a week from memory."
- **`CONTRIBUTING.md`** — the workflow rules (branch lifetime, tag conventions, review requirements, Day-0 checklist) live outside this document too, so a new contributor doesn't have to read the whole architecture spec to open a PR correctly.
- **Explicit post-freeze branch policy** — `hotfix/*` is the only branch type allowed to open after the Day-7 freeze tag.

---

## 1. Design principles (enforced, not just written down)

1. **Dataset replay only.** No live sensor, no real-time claim, ever. Every latency number is measured computational latency.
2. **NumPy end-to-end in the hot path.** No Open3D, no RANSAC, no Kalman/DBSCAN, no world-frame accumulation. `tests/architecture/` fails the build if any of these are imported anywhere under `src/fovmap/`.
3. **The map is robot-centric.** It lives in the current scan's frame; `T_rel` transforms the *stored map*, never the current scan.
4. **Re-binning (step 8) always precedes change detection (step 9).** This is structural — one function call order inside `pipeline/runner.py` — not a comment someone has to remember to respect.
5. **Every number in the final report traces to a runnable script and a manifest.** `scripts/trace_check.py` walks the report and confirms each figure resolves to a `manifest.json` with a git SHA and config hash — mechanically, not by eyeballing on Day 7.

---

## 2. Repository tree

```
foveated-lidar-mapping/
│
├── README.md                        # Setup, run instructions, known limitations
├── ARCHITECTURE.md                  # This document — tracked hand-edits only, no LLM regeneration
├── CONTRIBUTING.md                  # Branch lifetime, commit/tag conventions, review rules, Day-0 checklist
├── GATES.md                         # ★ append-only ledger: every gate pass, its tag, date, run manifest
├── pipeline.mmd                     # Mermaid source (version-controlled)
├── pipeline.html                    # Rendered flowchart (derived; regenerate from .mmd)
├── assets/
│   └── mermaid.min.js               # Vendored — pipeline.html must render with no internet
│
├── .gitignore
├── .pre-commit-config.yaml          # ruff format + lint on staged files, every commit
├── requirements.txt                 # torch cu128, numpy, streamlit, matplotlib, pandas — editable spec
├── requirements-lock.txt            # pip freeze, regenerated once, at the Day-7 freeze
├── pyproject.toml                   # installs src/fovmap as a real package — required, not optional
├── Makefile                         # setup / data / checkpoint / smoke / bench / dash / gate-dayN / lock
│
├── .github/
│   ├── CODEOWNERS                   # mechanical person → package ownership; drives PR auto-review
│   ├── PULL_REQUEST_TEMPLATE.md     # "which day-gate does this advance?"
│   └── workflows/
│       └── ci.yml                   # lint + CPU tests + architecture guards, every PR and push to main
│
├── data/                            # gitignored except the manifest
│   └── MANIFEST.md                  # seq-08 download (both archives), 7z extraction, sha256, delete zips
├── checkpoints/                     # gitignored except the manifest
│   └── MANIFEST.md                  # SalsaNext + RangeFormer checkpoints, sha256-verified, source URL
├── outputs/                         # gitignored — every run artifact
│   ├── predictions/                 #   per-scan LabeledCloud .npz (Day 2+)
│   ├── maps/                        #   MapSnapshot dump every ~10 scans (demo asset)
│   ├── bench/<run_id>/              #   manifest.json (git SHA + config hash) + claim tables
│   └── timing/                      #   headless component-latency JSONs
│
├── src/fovmap/                      # ONE installable package — no bare top-level module names
│   ├── __init__.py
│   │
│   ├── contracts.py                 # Every cross-module type, in one file. SCHEMA_VERSION-gated
│   │                                 # (see §3): ScanFrame/TRel, LabeledCloud, Cell/CellGrid,
│   │                                 # MapSnapshot, TimingRecord. This is what lets Person 6 mock,
│   │                                 # Person 4 go synthetic, and Persons 2/3 run raw samples on
│   │                                 # Day 1 — everyone codes against the same shapes from minute zero.
│   │
│   ├── data/                        # Person 1 — Steps 1, 2-input, 5
│   │   ├── loader.py                #   np.fromfile → (N,4) float32; iterates 4007 scans unattended
│   │   ├── labels.py                #   SemanticKITTI .label reader; semantic-kitti-api ordering
│   │   ├── remap.py                 #   19→4 table (DRIVABLE/TERRAIN/STATIC/OBJECT); Day-1 unit gate
│   │   ├── pose.py                  #   T_rel = Tr⁻¹·p_t⁻¹·p_{t-1}·Tr — current scan gets NO transform
│   │   ├── projection.py            #   NumPy→CUDA; 64×2048 spherical layout (row=ring, col=azimuth)
│   │   └── dataset.py               #   SemanticKITTISequence → yields (points, labels, T_rel)
│   │
│   ├── segmentation/                # Persons 2/3 — Steps 2-model, 3, 4
│   │   ├── salsanext/
│   │   │   ├── model.py             #   vendored 2020 architecture, torch-2.x patched
│   │   │   ├── VENDORED.md          #   upstream URL, pinned commit, patch list
│   │   │   ├── config.py
│   │   │   └── inference.py         #   checkpoint → (64×2048, 19) logits, single frame
│   │   ├── rangeformer/             #   FALLBACK — identical interface to salsanext/inference.py
│   │   ├── postprocess.py           #   median filter → exact pixel gather → confidence 19→4 remap
│   │   └── evaluate.py              #   mIoU + per-ring accuracy (subsample ~every 5th scan Day 1)
│   │
│   ├── grid/                        # Person 4 — Step 6 · CORE NOVELTY
│   │   ├── config.py                #   rings [10,25,50]m half-open; res [5,10,25,50]cm — integer
│   │   │                            #     multiples, essential for exact re-binning
│   │   ├── rings.py                 #   searchsorted([10,25,50], r, 'right'); boundary-deterministic
│   │   ├── indexing.py              #   (ring, floor(x/res), floor(y/res)) keys
│   │   ├── schema.py                #   Cell dataclass — mirrors contracts.Cell exactly
│   │   ├── engine.py                #   aggregate → CellGrid; asserts counts.sum()==N EVERY call
│   │   └── storage.py               #   sparse store + MapSnapshot dump every ~10 scans
│   │
│   ├── rating/                      # Person 5 — Steps 7, 8, 9
│   │   ├── traversability.py        #   ground = DRIVABLE cells' z_mean (NOT RANSAC); 20cm curb test
│   │   ├── dynamics.py              #   label disagreement + confidence + evidence gate; ghost decay
│   │   └── fusion.py                #   re-bin by T_rel → coarsen/refine → log-odds → prune >100m
│   │
│   ├── benchmarking/                # Person 5, + Persons 2/3 from Day 4 — central evidence / ML-ops
│   │   ├── baselines.py             #   uniform-5cm / uniform-50cm builders (reuses grid engine)
│   │   ├── harness.py               #   3 configs, same pipeline; memory/cells/latency each
│   │   ├── metrics.py               #   mIoU, P/R, false-drivable by 4 distance bands; ghost check
│   │   ├── latency.py               #   headless component-wise table — never during render
│   │   ├── report.py                #   central claim table assembly
│   │   ├── manifest.py              #   writes run_id = git SHA + config hash for every run
│   │   └── mlops.py                 #   Day-1 independent mIoU cross-check; TRT/FP16 (stretch)
│   │
│   ├── dashboard/                   # Person 6 — Step 10 · reads outputs/ only, never imports pipeline/
│   │   ├── app.py                   #   Streamlit entry: 5 views + performance panel
│   │   ├── views/
│   │   │   ├── semantic.py          #   real predictions, 4-group color code (Day-2 gate)
│   │   │   ├── elevation.py         #   real z stats (Day-4 gate)
│   │   │   ├── traversability.py    #   real scores (Day-4 gate)
│   │   │   ├── resolution_rings.py  #   4 ring bands from real grid (Day-3 gate)
│   │   │   └── dynamic_overlay.py   #   separate layer — dynamic cells excluded from persistent map
│   │   │                            #     but visible here, never silently dropped
│   │   ├── performance_panel.py     #   reads headless timing JSONs; never measures during render
│   │   ├── controls.py              #   uniform-vs-foveated toggle over saved outputs (Day-5 gate)
│   │   └── mock_data.py             #   contract-shaped fakes — Person 6 never blocks on anyone Day 1
│   │
│   └── pipeline/                    # Person 1 + Person 5 — shared wiring
│       ├── runner.py                #   THE per-scan loop. Re-bin BEFORE change-detect — structural.
│       ├── replay.py                #   simulated ~100ms replay cadence
│       └── timing.py                #   TimingRecord emission, consumed by benchmarking/latency.py
│
├── tests/
│   ├── conftest.py                  # fixtures: r=10/25/50 boundary points, synthetic 20cm curb,
│   │                                #   synthetic pose delta w/ hand-computed expected keys;
│   │                                #   registers `gpu` / `data` pytest markers
│   ├── unit/                        # test_remap · test_pose · test_ring_boundary · test_zero_loss ·
│   │                                #   test_zero_loss_real · test_traversability · test_rebinning ·
│   │                                #   test_schema_nonnull  (each named after its plan gate)
│   ├── integration/                 # test_loader_full_sequence · test_grid_real_data ·
│   │                                #   test_traversability_wired · test_10scan_coherence ·
│   │                                #   test_ghost_decay · test_full_pipeline
│   └── architecture/                # guardrails as code, not prose
│       ├── test_no_open3d.py            # open3d banned anywhere under src/fovmap/
│       ├── test_no_forbidden_deps.py    # RANSAC / sklearn-DBSCAN / filterpy-Kalman also banned
│       ├── test_grid_numpy_only.py      # grid/ must not import torch
│       ├── test_viz_decoupled.py        # dashboard/ must not be imported by pipeline/
│       ├── test_dag_respected.py        # full import-direction check — see §4
│       └── test_contracts_backward_compat.py  # Day-1 serialized fixture still loads today
│
├── scripts/
│   ├── validate_env.py              # Day-1: torch cu128 on Blackwell, seq 08 on disk, remap unit,
│   │                                #   checkpoint mIoU subsample check
│   ├── run_pipeline.py              # headless run; --seq 08 --config foveated|uniform5|uniform50
│   ├── run_dashboard.py             # Streamlit launch, separate process, reads outputs/ only
│   ├── run_benchmarks.py            # 3 configs → claim table + run manifests
│   ├── run_latency_headless.py      # component-wise latency table (no render)
│   └── trace_check.py               # Day-7 gate: every report number ↔ a bench manifest
│
└── docs/
    ├── plan.html                    # the original plan — tracked, supersedes all prior drafts
    ├── frame_convention.md          # Day-1 cam-vs-velo finding + conjugation formula
    ├── cell_schema.md               # field definitions + units
    ├── design_decisions.md          # what was cut and why (the guardrail reference)
    ├── benchmarking_protocol.md     # the number-traceability rule, spelled out
    ├── rebinning_caveats.md         # re-binning blur + rolling-memory semantics
    ├── resources.md                 # Consolidated Resource Table; "no invented links" rule —
    │                                #   an unverified URL is written as "search for X," never guessed
    └── risk_notes.md                # time-pressure rule, Person-5 load flag, Day-5 gate watch,
                                      #   repo hygiene, architecture guardrail warning
```

---

## 3. `contracts.py` — the integration backbone

Every type that crosses a module boundary lives in exactly one file. Nobody hand-rolls a dict shape and hopes the next module agrees with it.

```python
# src/fovmap/contracts.py
SCHEMA_VERSION = 1  # bump on any BREAKING change (see policy below)

@dataclass
class Cell:
    point_count: int
    z_mean: float
    z_min: float
    z_max: float
    semantic_label: int          # 4-group: DRIVABLE / TERRAIN / STATIC / OBJECT
    confidence: float
    evidence_count: int
    dynamic_flag: bool = False   # default → additive-safe
    timestamp: int = -1          # default → additive-safe
```

**Change policy** (this is what keeps "all-six review" from becoming a bottleneck):

| Change type | Example | Review | Version bump |
|---|---|---|---|
| **Additive** — new field with a default | Add `curvature: float = 0.0` to `Cell` | Single owner's PR, CI must pass | No |
| **Breaking** — rename, remove, retype a field | `z_mean: float` → `z_mean: np.float32` | All six, synchronous | Yes |

`tests/architecture/test_contracts_backward_compat.py` loads a `Cell` serialized on Day 1 against whatever `contracts.py` looks like today. If an additive change breaks that load, CI catches it before a human has to.

---

## 4. Module dependency graph — enforced, not just drawn

```
data ──► segmentation ──► grid ──► rating ──► benchmarking
  │            │            │         │             │
  └────────────┴────────────┴─────────┴─────────────┘
                         pipeline (orchestrator — imports downward only)

dashboard ──► contracts only, + reads outputs/*.npz off disk
              (never imports pipeline/, grid/, rating/, segmentation/, or data/ directly)
```

**Allowed imports, by package** (this table is the DAG's single source of truth, mirrored in `test_dag_respected.py`):

| Package | May import | May NOT import |
|---|---|---|
| `contracts` | stdlib, numpy | anything else in `fovmap` |
| `data` | `contracts` | everything else |
| `segmentation` | `contracts`, `data` | `grid`, `rating`, `benchmarking`, `dashboard`, `pipeline` |
| `grid` | `contracts` | `data`, `segmentation`, `rating`, `benchmarking`, `dashboard` |
| `rating` | `contracts`, `grid` | `data`, `segmentation`, `benchmarking`, `dashboard` |
| `benchmarking` | `contracts`, `grid`, `rating`, `segmentation` | `dashboard`, `pipeline` |
| `pipeline` | `contracts`, `data`, `segmentation`, `grid`, `rating` | `dashboard` |
| `dashboard` | `contracts` (+ filesystem reads of `outputs/`) | everything else |

`test_dag_respected.py` walks every `.py` file's AST, extracts its `import fovmap.X` statements, and fails the build on any edge not in this table. **Step 8-before-step-9 and "never render in the timed path" are instances of this one rule, not special-cased checks.**

---

## 5. Branching strategy

| Branch | Owner(s) | Opens | Lifetime | Merge condition | After merge |
|---|---|---|---|---|---|
| `main` | all | always exists | — | protected: CI green + auto-assigned CODEOWNERS review | — |
| `p1/<topic>` … `p6/<topic>` | matching person | fresh from `main` each morning | **≤ 1 day** | that day's gate passes → squash-merge | branch deleted |
| `stretch/trt-fp16` | Person 5 | only once **all** MIN gates for the day are green, team-wide | until Day 7 or abandoned | same CI rule | never blocks `main` |
| `stretch/rangeformer-compare` | Persons 2/3 | same condition | until Day 7 or abandoned | same CI rule | never blocks `main` |
| `hotfix/<topic>` | anyone | **only after the Day-7 freeze tag** | hours, not days | 2 approvals (freeze tightens review) | branch deleted |

**Why ≤1-day branches, not longer-lived ones:** the Day-4 gate ("real traversability + dynamic flags written into real cells") needs `rating/` and `grid/` both merged into `main` simultaneously. A branch that's still open from Day 2 makes that gate unsatisfiable. Daily merge cadence isn't a style preference here — it's what the schedule in §6 actually requires.

**Commit & tag conventions:**
- Prefix every commit `[P1]`…`[P6]` by owner; `[GATE dayN]` on the commit that closes a named gate; `[FREEZE]` on the Day-7 midday commit.
- Cut a **git tag** (`day1` … `day7`, `seg-freeze`, `v1.0.0`) at every gate pass — tags are immutable and checkout-able, commit-message tags are only searchable.
- After `[FREEZE]`: `main` becomes bugfix-only, two approvals required, and no new branch except `hotfix/*` may open.

---

## 6. Timeline — day gates

| Day | Gate focus | `make` target | Closes | Tag cut |
|---|---|---|---|---|
| **0** | Scaffold, not sprint — see §11 Day-0 checklist | `make setup` | Repo bootable by everyone | — |
| **1** | Env verified; ring boundaries deterministic; T_rel sanity; checkpoint mIoU in ballpark | `make gate-day1` | `validate_env.py` + `test_remap` + `test_pose` + `test_ring_boundary` | `day1` |
| **2** | Full-sequence loader; zero-loss on synthetic; per-ring accuracy printed | `make gate-day2` | `test_zero_loss` + `test_traversability` + `test_loader_full_sequence` | `day2` |
| **3** | Grid engine on real data; per-ring P/R table | `make gate-day3` | `test_grid_real_data` | `day3` |
| **4** | Cross-module wiring live; full sequence unattended | `make gate-day4` | `test_schema_nonnull` + `test_traversability_wired` | `day4` |
| **5** | Robot-centric fusion; 10-scan coherence; uniform-vs-foveated toggle | `make gate-day5` | `test_10scan_coherence` + `test_full_pipeline` | `day5` |
| **6** | Segmentation frozen; ghost-decay demo; headless latency table | `make gate-day6` | `test_ghost_decay` + `run_latency_headless.py` | `seg-freeze`, `day6` |
| **7 (midday)** | Freeze; every report number traced; fresh-env run confirmed | `make gate-day7` | `trace_check.py` + `make lock` + fresh-machine run | `[FREEZE]`, `v1.0.0` |

**Time-pressure rule (process, not code):** if a day's MIN gate isn't green by end of day, the next day starts by finishing it — `stretch/*` branches stay closed for that person until their MIN gate passes. `GATES.md` makes it obvious at a glance which gates are still open.

**`GATES.md` format** (appended by every `make gate-dayN` on success, never hand-edited):

```
## day3 — 2026-MM-DD 18:42 UTC
tag: day3
commits: a1b2c3d..e4f5a6b
tests: test_grid_real_data — PASS
notes: per-ring P/R table at outputs/bench/day3-eval/report.md
```

---

## 7. Usage guide

### Makefile targets

| Command | Does | Who runs it | When |
|---|---|---|---|
| `make setup` | Creates venv, installs `fovmap` editable, installs pre-commit hooks | everyone | Day 0, once |
| `make data` | Runs `data/MANIFEST.md` download + selective 7z extraction, verifies sha256 | Person 1 | Day 1 |
| `make checkpoint` | Downloads + sha256-verifies SalsaNext (and RangeFormer if triggered) | Persons 2/3 | Day 1 |
| `make smoke` | `scripts/run_pipeline.py` on a handful of scans, no render | anyone | Daily, before pushing |
| `make bench` | `scripts/run_benchmarks.py`, all 3 grid configs, writes `outputs/bench/<run_id>/manifest.json` | Person 5 (+ P2/3 from Day 4) | Days 3–7 |
| `make dash` | Launches the Streamlit dashboard against `outputs/` | Person 6 | Daily |
| `make gate-dayN` | Runs that day's test subset, appends to `GATES.md`, prints the tag command | whoever owns the day's last gate | End of each day |
| `make lock` | Regenerates `requirements-lock.txt` from the current venv | Person 1 | Day 7 only |

### Typical workflows

**Starting your day (any person, Days 1–6):**
```
git checkout main && git pull
git checkout -b p4/rings-boundary-fix
# ... work ...
git push -u origin p4/rings-boundary-fix   # opens PR, CI runs, CODEOWNERS auto-assigned
```
Squash-merge once CI is green and the day's gate test passes locally. Delete the branch. Re-branch fresh tomorrow.

**Adding a field to a contract (additive):**
1. Add the field to the relevant dataclass in `contracts.py` **with a default value**.
2. Update any producer/consumer that should populate it.
3. `pytest tests/architecture/test_contracts_backward_compat.py` locally — must still pass.
4. Open a normal PR; single-owner review is enough.

**Adding a field to a contract (breaking — renaming or retyping):**
1. Bump `SCHEMA_VERSION`.
2. Message all six co-owners before opening the PR — this one needs synchronous review, not async.
3. Note the migration in a comment at the top of `contracts.py`.

**Running a benchmark whose numbers will go in the report:**
```
make bench CONFIG=foveated
```
Writes to `outputs/bench/<run_id>/` with `manifest.json` (git SHA + config hash). Every number quoted in the final report must come from a directory like this — `scripts/trace_check.py` checks it on Day 7.

**Cutting a day's gate:**
```
make gate-day5
git tag day5 && git push --tags
```
The Makefile target itself appends the `GATES.md` entry; the tag is a separate, deliberate step so a green test run doesn't silently become "the gate passed" without someone confirming it.

**Freeze day:**
```
make gate-day7      # runs trace_check.py against the full report
make lock            # requirements-lock.txt
git tag v1.0.0 && git push --tags
```
From this point, `main` requires two approvals and only `hotfix/*` branches may open.

---

## 8. Enforcement

**`.gitignore`** (fixed: ignores dataset/checkpoint *contents*, keeps the tracked manifests):
```gitignore
data/*
!data/MANIFEST.md
checkpoints/*
!checkpoints/MANIFEST.md

outputs/
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.ipynb_checkpoints/
.DS_Store
```

**`.github/workflows/ci.yml`:**
```yaml
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: ruff check src tests scripts
      - run: pytest -m "not gpu and not data"   # unit + architecture guards only
```
The `gpu`/`data` markers (registered in `conftest.py`) keep CI CPU-only and green; anything needing the Blackwell box or the full dataset runs locally via `make smoke` / `make bench`. Don't burn sprint time chasing CI flakiness on GPU jobs.

**`.github/CODEOWNERS`:**
```
/src/fovmap/data/           @person1
/src/fovmap/segmentation/   @person2 @person3
/src/fovmap/grid/           @person4
/src/fovmap/rating/         @person5
/src/fovmap/benchmarking/   @person5 @person2 @person3
/src/fovmap/dashboard/      @person6
/src/fovmap/pipeline/       @person1 @person5
/src/fovmap/contracts.py    @person1 @person2 @person3 @person4 @person5 @person6
```

**`tests/architecture/` — the guardrails, as running code:**
| Test | Guards against |
|---|---|
| `test_no_open3d.py` | Open3D anywhere under `src/fovmap/` |
| `test_no_forbidden_deps.py` | RANSAC libs, sklearn DBSCAN, filterpy/Kalman imports |
| `test_grid_numpy_only.py` | `grid/` importing `torch` |
| `test_viz_decoupled.py` | `pipeline/` importing `dashboard/` |
| `test_dag_respected.py` | any import edge outside the §4 table |
| `test_contracts_backward_compat.py` | an additive `contracts.py` change breaking old serialized data |

**`.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

---

## 9. `docs/` index

| File | Purpose |
|---|---|
| `plan.html` | The original plan spec — tracked, supersedes all prior drafts |
| `frame_convention.md` | Day-1 camera-vs-Velodyne finding + the conjugation formula |
| `cell_schema.md` | Cell field definitions and units |
| `design_decisions.md` | What was cut (PointNet++, Open3D, RANSAC, Kalman/DBSCAN, world-frame) and why |
| `benchmarking_protocol.md` | The number-traceability rule, spelled out for report writers |
| `rebinning_caveats.md` | Re-binning blur + what "the map" does and doesn't mean (bounded rolling memory) |
| `resources.md` | Every paper/repo/doc/tool referenced by the plan, one place, "no invented links" |
| `risk_notes.md` | Time-pressure rule, Person-5 load-bearing flag, Day-5 gate warning, repo hygiene, architecture guardrail reminder |

---

## 10. Data & checkpoint manifests

```
data/MANIFEST.md
  - Source: semantic-kitti.org (labels) + KITTI odometry (points), both archives required
  - Extraction: selective 7-Zip, sequence 08 only (4007 scans)
  - sha256 of each extracted archive recorded here before the zips are deleted
  - Post-extraction: zips deleted to reclaim disk

checkpoints/MANIFEST.md
  - SalsaNext: search "SalsaNext GitHub Cortinhal," verify the link resolves on Day 1, record sha256
  - RangeFormer (fallback): same verify-then-record process
  - Never printed as an unverified URL — if unconfirmed, this file says "search for X"
```

---

## 11. Day-0 checklist (before Day 1 starts)

- [ ] `contracts.py` drafted together, all six present — this is the file everyone's Day-1 work depends on existing in at least stub form.
- [ ] `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `.github/CODEOWNERS` committed to `main`.
- [ ] Branch protection turned on for `main` (CI required, review required).
- [ ] Every person runs `make setup` locally and confirms it succeeds.
- [ ] `assets/mermaid.min.js` vendored; `pipeline.html` confirmed to render with no network connection.

---

## 12. Role → ownership quick reference

| Person | Owns | Day-1 deliverable everyone else depends on |
|---|---|---|
| **1** | `data/`, co-owns `pipeline/` | Clean current-frame point cloud + `T_rel`, at least in stub form |
| **2 / 3** | `segmentation/` | A working SalsaNext (or RangeFormer fallback) checkpoint at paper-ballpark mIoU |
| **4** | `grid/` (core novelty) | Deterministic ring-boundary function — needs no network, starts immediately |
| **5** | `rating/`, co-owns `pipeline/`, leads `benchmarking/` | Draft elevation-stats function against synthetic cells; independent mIoU cross-check |
| **6** | `dashboard/` | Streamlit skeleton against `mock_data.py` — never blocks waiting on anyone |
| **All six** | `contracts.py` | The shared types that make Day-1 parallel work safe rather than lucky |
