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

Tomography and planning share one scene definition under `pct_config/scenes/`. All scene names use lowercase snake case. The available scenes are:

```text
building
gx_forest_gg2
gx_forest_gg3
plaza
shenkan_mls
spiral
whu_tls_forest
```

Each `scene_<name>.py` owns all scene-specific configuration:

- `SceneMap(...)` and `SceneTrav(...)` for tomogram construction;
- PCD input, tomogram output/input, and waypoint output paths;
- one or more named routes, each with a start, end, and waypoint filename;
- `waypoint_offset=(x, y, z)`, applied only when TXT/PCD waypoints are written. Existing scenes use `(0.0, 0.0, 1.6)`.

`--scene <name>` is resolved dynamically to `pct_config/scenes/scene_<name>.py`; the two main programs contain no scene-specific branches.

### Adding a Scene

Copy `pct_config/templates/scene_height_only.py` to `pct_config/scenes/scene_<name>.py`, then fill in `SCENE`. The filename, `SCENE.name`, and CLI value must use the same lowercase name. No registry or main-program edit is needed: a valid module is discovered automatically.

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

The refactored configuration was validated in the prepared `enter-pct_planner` environment using the full 15,095,947-point WHU-TLS cloud and all five routes. The tomogram SHA-256 is identical to the current-code baseline. All five generated TXT trajectories are numerically identical to that baseline. Compared with the older `data/path_planning` outputs, point counts and XY are identical; routes 2-5 retain the documented `+0.6 m` Z difference because the scene keeps the historical `(0.0, 0.0, 1.6)` waypoint offset, and route 1 contains 28 older manual height edits.

Exact commands, hashes, artifacts, and environment caveats are recorded in `AI_prompt/README.md` and the simulation result report.

## License

The source code is released under [GPLv2](http://www.gnu.org/licenses/) license. For commercial use, contact Bowen Yang at [byangar@connect.ust.hk](mailto:byangar@connect.ust.hk).

## 固定位置项目环境（附加说明）

Python/CUDA、GTSAM、OSQP 和 native modules 的首次环境建立，
`enter-pct_planner` 身份检查、源码增量编译、运行与产物可删除性统一见
[`INDEPENDENT_ENVIRONMENT.md`](INDEPENDENT_ENVIRONMENT.md)。长命令不在 README 与
AI 导航中重复维护；原生 CMake 没有项目 defaults 文件，profile 使用有效的环境
变量和标准命令。
