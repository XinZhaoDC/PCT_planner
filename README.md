# PCT Planner

## Overview

This is an implementation of the paper **Efficient Global Navigational Planning in 3D Structures based on Point Cloud Tomography** (submitted to TMECH). It provides a highly efficient and extensible global navigation framework based on a tomographic understanding of the environment to navigate ground robots in multi-layer structures.

**Demonstrations**: [pct_planner](https://byangw.github.io/projects/tmech2024/)

![demo](rsc/docs/demo.png)

## Prerequisites

- Ubuntu >= 20.04
- ROS >= Noetic with ros-desktop-full installation
- CUDA >= 11.7
- Python >= 3.8
- [CuPy](https://docs.cupy.dev/en/stable/install.html) with CUDA >= 11.7
- Open3D

## Build & Install

The package has two modules: point cloud tomography in `tomography/` and path planning/optimization in `planner/`. Only the planner module needs to be built before use.

```bash
cd planner
./build_thirdparty.sh
./build.sh
```

Those scripts are the upstream source-local workflow. In this container's
`enter-pct_planner` profile, use
[`INDEPENDENT_ENVIRONMENT.md`](INDEPENDENT_ENVIRONMENT.md) instead; the isolated
workflow does not run scripts that remove source-local third-party build/install
trees.

## Scene Configuration

Tomography and planning share one scene definition under `pct_config/scenes/`.
All scene names use lowercase snake case. The available scenes are:

```text
building
gx_forest_gg2
gx_forest_gg3
plaza
shenkan_mls
spiral
whu_tls_forest
```

`--scene <name>` is resolved dynamically to
`pct_config/scenes/scene_<name>.py`; the tomography and planning entry points
contain no scene-specific branches. Each module exports one immutable
`SceneConfig` named `SCENE`.

### SceneConfig fields

| Field | Meaning |
|---|---|
| `name` | Lowercase scene identifier. It must match the module filename and the value passed to `--scene`. |
| `pcd_file` | Source point cloud read by `tomography.py`. Relative paths are resolved from the repository root; absolute paths are preserved. |
| `tomogram_file` | Pickle written by `tomography.py` and read by `plan.py`. Both stages must use the same file. |
| `waypoint_dir` | Directory where `plan.py` writes route TXT files and optional binary PCD files. |
| `waypoint_offset` | `(x, y, z)` translation added after planning, immediately before TXT/PCD export. It does not change the optimizer, planner-internal trajectory, configured `RouteConfig.start/end`, or published ROS Path; it does translate the coordinates, including the first and last points, stored in the exported files. |
| `map` | `SceneMap` parameters controlling horizontal discretization and vertical slicing during tomography. |
| `trav` | `SceneTrav` parameters controlling clearance, terrain, obstacle-cost, and inflation calculations. |
| `routes` | Mapping from CLI route names to `RouteConfig` objects. The reserved CLI value `all` cannot be used as a route name. |
| `default_route` | Route selected when `--route` is omitted; it must exist in `routes`. |

Every formal scene currently sets `waypoint_offset` explicitly. The
`SceneConfig` dataclass value `(0.0, 0.0, 1.6)` is only a construction fallback
for callers that omit the field; it is not a universal offset for all datasets.
The effective values are scene-specific:

| Scene | Effective `waypoint_offset` |
|---|---|
| `whu_tls_forest` | `(0.0, 0.0, 1.0)` |
| `building`, `gx_forest_gg2`, `gx_forest_gg3`, `plaza`, `shenkan_mls`, `spiral` | `(0.0, 0.0, 1.6)` |

Choose an offset from the coordinate convention of the dataset and its
consumer. In particular, the WHU value `1.0` reproduces the retained old five
route PCDs; changing it to `1.6` raises every exported WHU waypoint by about
`0.6 m` without changing XY or the planner result.

### SceneMap fields

All distances are in metres unless stated otherwise.

| Field | Meaning |
|---|---|
| `resolution` | Horizontal grid resolution in metres per cell. It controls point-cloud rasterization and world/grid conversion. |
| `ground_h` | Z value used as the lower bound of the scene before vertical slices are created. The first slice is anchored at `ground_h + slice_dh`. |
| `slice_dh` | Vertical spacing between adjacent tomogram slices. |

### SceneTrav fields

| Field | Meaning |
|---|---|
| `kernel_size` | Nominal local footprint width in grid cells used to examine neighbouring standable cells. The implementation converts it to an integer half-width. |
| `interval_min` | Minimum vertical clearance. Cells below this clearance receive `cost_barrier`. |
| `interval_free` | Clearance at or above which no clearance penalty is added; values between `interval_min` and `interval_free` receive a graded penalty. |
| `slope_max` | Slope-angle threshold in radians used to derive the standable step threshold `1.2 * resolution * tan(slope_max)`. |
| `step_max` | Maximum local height step considered crossable. Larger steps are barriers. |
| `standable_ratio` | Required proportion of standable cells in the local footprint for terrain that is crossable but not directly standable. |
| `cost_barrier` | Traversability cost assigned to obstacles and rejected cells. |
| `safe_margin` | Additional distance over which an inflated obstacle cost decays. |
| `inflation` | Radius around an obstacle that retains full inflation weight before the `safe_margin` decay. |

### RouteConfig fields

| Field | Meaning |
|---|---|
| `start`, `end` | Two-dimensional world coordinates passed to the planner. Their grid-snapped trajectory endpoints can differ slightly at `resolution` precision. |
| `waypoint_file` | Output stem; `plan.py` writes `<waypoint_file>.txt` and, with `--write-pcd`, `<waypoint_file>.pcd`. |
| `include_in_all` | Whether `--route all` includes this route. Use `False` for compatibility/default aliases that should not be duplicated in a batch. |

### Adding a Scene

Copy `pct_config/templates/scene_height_only.py` to
`pct_config/scenes/scene_<name>.py`, then fill in every `SCENE` field. Treat the
template values, including `waypoint_offset`, as examples rather than dataset
defaults. The filename, `SCENE.name`, and CLI value must use the same lowercase
name. No registry or main-program edit is needed: a valid module is discovered
automatically.

Use a `default` route for normal `plan.py --scene <name>` execution. Additional route names can be selected using `--route <route>`; `--route all` runs every route whose `include_in_all` is true.

## Running

Start `roscore` before commands that publish ROS messages. In this workspace, enter the prepared environment first:

```bash
enter-pct_planner
```

### Tomogram Construction

Unzip `rsc/pcd/pcd_files.zip` into `rsc/pcd/`, or set an explicit PCD path. For example:

```bash
cd tomography/scripts
python3 tomography.py --scene spiral
```

The default PCD and tomogram paths come from `pct_config/scenes/scene_spiral.py`. They can be overridden without changing the scene file:

```bash
python3 tomography.py \
  --scene whu_tls_forest \
  --pcd-file /path/to/WHU_TLS_Forest.pcd \
  --tomogram-file /path/to/output/WHU_TLS_Forest.pickle \
  --no-publish \
  --no-spin
```

### Trajectory Generation

```bash
cd planner/scripts
python3 plan.py --scene spiral
```

`enter-pct_planner` already provides the Python, native-module, GTSAM, OSQP,
CUDA userspace, ROS Python, and Ubuntu dist-packages paths required by the
planner. Do not append a source-local GTSAM install path manually.

Run one named WHU route or all five routes:

```bash
python3 plan.py \
  --scene whu_tls_forest \
  --route all \
  --tomogram-file /path/to/WHU_TLS_Forest.pickle \
  --output-dir /path/to/waypoint \
  --write-pcd \
  --overwrite \
  --no-spin
```

`--overwrite` replaces existing TXT output; without it the historical append behavior is retained. `--write-pcd` additionally writes the binary PCL PointXYZ format used by Marsim.

## WHU-TLS Validation

The shared-scene refactor was validated in the prepared `enter-pct_planner`
environment using the full 15,095,947-point WHU-TLS cloud and all five routes.
The retained tomogram SHA-256 is
`b7efbfbe59568009656fbcbb32f1146e9ccf9a2f48b3bcc5eb2c4bc1d1e3d507`.

The export offset was audited again on 2026-10-08. With
`waypoint_offset=(0.0, 0.0, 1.0)`, the five generated PCDs contain
496/541/451/379/352 points and are byte-for-byte identical to the corresponding
retained old WHU route files. With `z=1.6`, every route keeps the same point
count, order, and XY but every exported Z is about `0.6 m` higher. The active
Route 1 file also contains 28 historical manual Z edits; those edits are a
separate consumer-data convention and are not part of planner reproduction.

Exact commands, hashes, artifacts, and environment caveats are recorded in
`AI_prompt/README.md` and
`/home/workspace/simulation/.ai/results/HO-20261007-004-result.md`.

## License

The source code is released under [GPLv2](http://www.gnu.org/licenses/) license. For commercial use, contact Bowen Yang at [byangar@connect.ust.hk](mailto:byangar@connect.ust.hk).

## 固定位置项目环境（附加说明）

Python/CUDA、GTSAM、OSQP 和 native modules 的首次环境建立，
`enter-pct_planner` 身份检查、源码增量编译、运行与产物可删除性统一见
[`INDEPENDENT_ENVIRONMENT.md`](INDEPENDENT_ENVIRONMENT.md)。长命令不在 README 与
AI 导航中重复维护；原生 CMake 没有项目 defaults 文件，profile 使用有效的环境
变量和标准命令。
