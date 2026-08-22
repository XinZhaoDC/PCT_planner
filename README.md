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

上面的正常构建与运行方式继续保留。以下命令在固定位置创建 Python 3.8/CUDA 11.8 环境；环境入口脚本由挂载根统一维护，不在仓库 README 中展开其内容。

```bash
PCT_PREFIX=/home/workspace/simulation/environment/prefixes/pct_planner-noetic-py38-cu118
python3 -m venv "$PCT_PREFIX"
source "$PCT_PREFIX/bin/activate"

python -m pip install --no-index \
  --find-links=/home/workspace/simulation/download/pct_planner_py38 \
  numpy==1.24.4 cupy-cuda11x==12.3.0 fastrlock==0.8.3 \
  nvidia-cuda-runtime-cu11==11.8.89 \
  nvidia-cuda-nvrtc-cu11==11.8.89 \
  nvidia-cublas-cu11==11.11.3.6 \
  nvidia-cufft-cu11==10.9.0.58 \
  nvidia-curand-cu11==10.3.0.86 \
  nvidia-cusolver-cu11==11.4.1.48 \
  nvidia-cusparse-cu11==11.7.5.86

PCT_REPO=/home/workspace/simulation/code/pct_planner/src/PCT_planner
PCT_BUILD=/home/workspace/simulation/environment/build/pct_planner-noetic-py38-cu118
PCT_VIEW="$PCT_BUILD/source-view"
PCT_NATIVE="$PCT_PREFIX/native"
PCT_PYTHON="$PCT_PREFIX/native-python"

mkdir -p "$PCT_VIEW" "$PCT_NATIVE/gtsam-4.1.1" \
  "$PCT_NATIVE/osqp-0.6.2" "$PCT_PYTHON/lib"
tar --exclude='./build' \
  --exclude='./3rdparty/gtsam-4.1.1/build' \
  --exclude='./3rdparty/gtsam-4.1.1/install' \
  --exclude='./3rdparty/osqp/build' \
  --exclude='./3rdparty/osqp/install' \
  --exclude='*.so' --exclude='*.so.*' \
  -C "$PCT_REPO/planner/lib" -cf - . | tar -C "$PCT_VIEW" -xf -
mkdir -p "$PCT_VIEW/3rdparty/gtsam-4.1.1" "$PCT_VIEW/3rdparty/osqp"
ln -s "$PCT_NATIVE/gtsam-4.1.1" "$PCT_VIEW/3rdparty/gtsam-4.1.1/install"
ln -s "$PCT_NATIVE/osqp-0.6.2" "$PCT_VIEW/3rdparty/osqp/install"

cmake -S "$PCT_VIEW/3rdparty/gtsam-4.1.1" -B "$PCT_BUILD/gtsam-4.1.1" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PCT_NATIVE/gtsam-4.1.1" \
  -DBUILD_SHARED_LIBS=ON -DGTSAM_BUILD_UNSTABLE=OFF \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_PYTHON=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=ON \
  -DGTSAM_USE_SYSTEM_EIGEN=ON
cmake --build "$PCT_BUILD/gtsam-4.1.1" --target install --parallel 2

cmake -S "$PCT_VIEW/3rdparty/osqp" -B "$PCT_BUILD/osqp-0.6.2" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PCT_NATIVE/osqp-0.6.2" \
  -DUNITTESTS=OFF -DPRINTING=OFF
cmake --build "$PCT_BUILD/osqp-0.6.2" --target install --parallel 2

cmake -S "$PCT_VIEW" -B "$PCT_BUILD/native-modules" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$PCT_PREFIX/bin/python" \
  -DPython_EXECUTABLE="$PCT_PREFIX/bin/python"
cmake --build "$PCT_BUILD/native-modules" --parallel 2
cp "$PCT_BUILD"/native-modules/src/a_star/a_star*.so "$PCT_PYTHON/lib/"
cp "$PCT_BUILD"/native-modules/src/trajectory_optimization/traj_opt*.so "$PCT_PYTHON/lib/"
cp "$PCT_BUILD"/native-modules/src/ele_planner/ele_planner*.so "$PCT_PYTHON/lib/"
cp "$PCT_BUILD"/native-modules/src/map_manager/py_map_manager*.so "$PCT_PYTHON/lib/"
```

实际运行 tomography 和规划：

```bash
source /home/workspace/simulation/environment/shell_entries.sh
enter-pct_planner
roscore

# 在另一个 enter-pct_planner 终端中运行：
cd /home/workspace/simulation/code/pct_planner/src/PCT_planner/tomography/scripts
python tomography.py \
  --scene whu_tls_forest \
  --pcd-file /home/workspace/simulation/data/path_planning/scene/WHU_TLS_Forest.pcd \
  --tomogram-file /home/workspace/simulation/data/AI_tests/pct/tomogram/WHU_TLS_Forest.pickle \
  --no-publish --no-spin

cd /home/workspace/simulation/code/pct_planner/src/PCT_planner/planner/scripts
python plan.py \
  --scene whu_tls_forest --route 1 \
  --tomogram-file /home/workspace/simulation/data/AI_tests/pct/tomogram/WHU_TLS_Forest.pickle \
  --output-dir /home/workspace/simulation/data/AI_tests/pct/waypoint \
  --write-pcd --overwrite --no-spin
```

运行结束使用 `exit` 返回原 shell。相同的固定位置创建、编译和运行引导也保存在 [`AI_prompt/README.md`](AI_prompt/README.md)。
