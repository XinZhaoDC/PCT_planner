# PCT Planner 独立隔离环境

本文是 PCT Planner 固定位置 Python/CUDA/native 环境的权威指南。仓库根 README
保留算法接口；依赖环境建立、`enter-pct_planner`、源码重编译、运行和产物生命周期
统一在这里维护。

## 1. 环境边界

- Profile：`pct-noetic-py38`
- 仓库：`/home/workspace/simulation/code/pct_planner/src/PCT_planner`
- Python/runtime prefix：`/home/workspace/simulation/environment/prefixes/pct_planner-noetic-py38-cu118`
- 隔离 build 根：`/home/workspace/simulation/environment/build/pct_planner-noetic-py38-cu118`
- 依赖：CPython 3.8、NumPy 1.24.4、CuPy 12.3.0、CUDA 11.8 userspace wheels、GTSAM 4.1.1、OSQP 0.6.2、ROS Noetic
- 构建工具：保持原生 CMake/native module 模型，不改成 colcon

`setup.sh` 只激活已安装环境。它导出 `PCT_REPO`、`PCT_SOURCE`、
`PCT_SOURCE_VIEW`、`PCT_BUILD`、`PCT_NATIVE_BUILD`、`PCT_PREFIX`、
`PCT_NATIVE`、`PCT_PYTHON`、`PCT_SITE`、`PCT_COPY_TARGET`、`PCT_BUILD_JOBS` 和
`PCT_BUILD_TYPE`。profile
通过 CMake 原生的 `CMAKE_BUILD_PARALLEL_LEVEL=2` 提供默认并行度。原生 CMake 没有
项目 defaults-file 接口；其他配置通过稳定变量和标准命令显式给出，不增加不会被读取
的 YAML、自定义 build helper 或与 `PCT_NATIVE` 同值的 install 别名。四个扩展复制到
`PCT_COPY_TARGET`，该精确目录同时进入 `PYTHONPATH`。

## 2. 首次构建环境

本节篇幅最大，因为 Python ABI、CUDA userspace、GTSAM、OSQP、pybind 模块和 ROS
Python 必须同时一致。所有安装和首次构建由用户手工分批执行。已有 prefix 或 build
树与本文不一致时停止并保留现场，不覆盖当前源码、不在线下载，也不运行仓库内会先
删除 `build/install` 的旧 `build_thirdparty.sh`。

### 2.1 只读预检

```bash
test -r /opt/ros/noetic/setup.bash
test -d /home/workspace/simulation/download/pct_planner_py38
test -f /home/workspace/Forest_Interface/download/python-wheels/open3d-0.18.0-cp38-cp38-manylinux_2_27_x86_64.whl
test -d /home/workspace/simulation/code/pct_planner/src/PCT_planner/planner/lib
python3 --version
df -h /home/workspace/simulation
```

首次创建前还应确认目标不存在：

```bash
test ! -e /home/workspace/simulation/environment/prefixes/pct_planner-noetic-py38-cu118
test ! -e /home/workspace/simulation/environment/build/pct_planner-noetic-py38-cu118
```

若目标已存在，这不是错误清理信号，而是“复用、修复或新建版本化 profile”的决策门。

### 2.2 创建 Python 3.8 prefix

系统 `python3` 必须确认是 CPython 3.8 后再创建 venv。所有 wheel 从保留的本地目录
安装，不访问网络：

```bash
PCT_PREFIX=/home/workspace/simulation/environment/prefixes/pct_planner-noetic-py38-cu118
python3 -m venv "$PCT_PREFIX"
source "$PCT_PREFIX/bin/activate"
python -m pip install --no-index \
  --find-links=/home/workspace/simulation/download/pct_planner_py38 \
  --find-links=/home/workspace/Forest_Interface/download/python-wheels \
  numpy==1.24.4 \
  cupy-cuda11x==12.3.0 \
  fastrlock==0.8.3 \
  nvidia-cuda-runtime-cu11==11.8.89 \
  nvidia-cuda-nvrtc-cu11==11.8.89 \
  nvidia-cublas-cu11==11.11.3.6 \
  nvidia-cufft-cu11==10.9.0.58 \
  nvidia-curand-cu11==10.3.0.86 \
  nvidia-cusolver-cu11==11.4.1.48 \
  nvidia-cusparse-cu11==11.7.5.86
python -c 'import sys, numpy; print(sys.executable, sys.version, numpy.__version__)'
```

NVIDIA kernel driver 由容器运行时提供，不复制进 prefix。CuPy wheel 可安装成功但 GPU
仍可能不可用，因此安装成功不等于运行验收成功。

### 2.3 建立隔离 source view

原仓库 CMake 期望 `3rdparty/*/install` 位于源码视图内。为避免在 Git 工作树生成
依赖 install 和 `.so`，固定环境把 `planner/lib` 复制到 build 根的 source view，
并把两个 install 路径链接到持久 native prefix：

```bash
PCT_REPO=/home/workspace/simulation/code/pct_planner/src/PCT_planner
PCT_BUILD=/home/workspace/simulation/environment/build/pct_planner-noetic-py38-cu118
PCT_VIEW="$PCT_BUILD/source-view"
PCT_NATIVE="$PCT_PREFIX/native"
PCT_PYTHON="$PCT_PREFIX/native-python"

mkdir -p "$PCT_VIEW" \
  "$PCT_NATIVE/gtsam-4.1.1" \
  "$PCT_NATIVE/osqp-0.6.2" \
  "$PCT_PYTHON/lib"
tar --exclude='./build' \
  --exclude='./3rdparty/gtsam-4.1.1/build' \
  --exclude='./3rdparty/gtsam-4.1.1/install' \
  --exclude='./3rdparty/osqp/build' \
  --exclude='./3rdparty/osqp/install' \
  --exclude='*.so' --exclude='*.so.*' \
  -C "$PCT_REPO/planner/lib" -cf - . | tar -C "$PCT_VIEW" -xf -
mkdir -p "$PCT_VIEW/3rdparty/gtsam-4.1.1" "$PCT_VIEW/3rdparty/osqp"
ln -s "$PCT_NATIVE/gtsam-4.1.1" \
  "$PCT_VIEW/3rdparty/gtsam-4.1.1/install"
ln -s "$PCT_NATIVE/osqp-0.6.2" \
  "$PCT_VIEW/3rdparty/osqp/install"
```

若 link 已存在，先用 `readlink -f` 验证；不要用强制覆盖参数。
首次 source-view 应复制仓库中的 OSQP 配置头模板。随后第 2.5 节配置 OSQP 时，
其 CMake 会按本环境的 `PRINTING=OFF` 等选项更新
`$PCT_VIEW/3rdparty/osqp/include/osqp_configure.h`；更新后的文件属于该 source-view
的已配置状态，不应在日常源码增量同步中被仓库模板覆盖。

### 2.4 构建并安装 GTSAM 4.1.1

```bash
cmake -S "$PCT_VIEW/3rdparty/gtsam-4.1.1" \
  -B "$PCT_BUILD/gtsam-4.1.1" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PCT_NATIVE/gtsam-4.1.1" \
  -DBUILD_SHARED_LIBS=ON \
  -DGTSAM_BUILD_UNSTABLE=OFF \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_PYTHON=OFF \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=ON \
  -DGTSAM_USE_SYSTEM_EIGEN=ON
cmake --build "$PCT_BUILD/gtsam-4.1.1" --target install --parallel 2
test -r "$PCT_NATIVE/gtsam-4.1.1/lib/cmake/GTSAM/GTSAMConfig.cmake"
```

### 2.5 构建并安装 OSQP 0.6.2

```bash
cmake -S "$PCT_VIEW/3rdparty/osqp" \
  -B "$PCT_BUILD/osqp-0.6.2" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PCT_NATIVE/osqp-0.6.2" \
  -DUNITTESTS=OFF \
  -DPRINTING=OFF
cmake --build "$PCT_BUILD/osqp-0.6.2" --target install --parallel 2
test -r "$PCT_NATIVE/osqp-0.6.2/lib/cmake/osqp/osqp-config.cmake"
```

### 2.6 构建并复制 native modules

```bash
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

四个 Python 扩展复制到固定 runtime 目录；`libcommon_smoothing.so` 等关联库仍从
`native-modules` build tree 加载。这是当前已验证模型，不能把整个 native build tree
误判为可随时删除的纯缓存。

### 2.7 Profile 文件

固定 profile 至少包含：

```text
/home/workspace/simulation/environment/profiles/pct-noetic-py38/setup.sh
/home/workspace/simulation/environment/profiles/pct-noetic-py38/enter.sh
/home/workspace/simulation/environment/profiles/pct-noetic-py38/versions.lock
```

Profile 激活不得执行上述安装或构建步骤。

## 3. 进入与身份检查

```bash
source /home/workspace/simulation/environment/shell_entries.sh
enter-pct_planner
printf 'profile=%s\npython=%s\nrepo=%s\nsource-view=%s\nbuild=%s\ncopy=%s\n' \
  "$SIMULATION_ENV_PROFILE" "$(command -v python)" "$PCT_REPO" \
  "$PCT_SOURCE_VIEW" "$PCT_NATIVE_BUILD" "$PCT_COPY_TARGET"
```

预期 Python 位于 `PCT_PREFIX/bin`，路径均属于 simulation 根。用独立 Python
进程分别探测 native module，避免重复加载造成假故障：

```bash
python -c 'import a_star; print(a_star.__file__)'
python -c 'import traj_opt; print(traj_opt.__file__)'
python -c 'import ele_planner; print(ele_planner.__file__)'
python -c 'import py_map_manager; print(py_map_manager.__file__)'
python -c 'import cupy; print(cupy.__version__)'
```

## 4. 源码增量编译

仓库中的 `planner/lib` 与隔离 `source-view` 是两棵树。修改 C++/pybind 源码后，先
同步非构建文件，再重新配置和编译：

```bash
tar --exclude='./build' \
  --exclude='./3rdparty/gtsam-4.1.1/build' \
  --exclude='./3rdparty/gtsam-4.1.1/install' \
  --exclude='./3rdparty/osqp/build' \
  --exclude='./3rdparty/osqp/install' \
  --exclude='./3rdparty/osqp/include/osqp_configure.h' \
  --exclude='*.so' --exclude='*.so.*' \
  -C "$PCT_SOURCE" -cf - . | tar -C "$PCT_SOURCE_VIEW" -xf -
cmake -S "$PCT_SOURCE_VIEW" -B "$PCT_NATIVE_BUILD" \
  -DCMAKE_BUILD_TYPE="$PCT_BUILD_TYPE" \
  -DPYTHON_EXECUTABLE="$PCT_PREFIX/bin/python" \
  -DPython_EXECUTABLE="$PCT_PREFIX/bin/python"
cmake --build "$PCT_NATIVE_BUILD"
cp "$PCT_NATIVE_BUILD"/src/a_star/a_star*.so "$PCT_COPY_TARGET/"
cp "$PCT_NATIVE_BUILD"/src/trajectory_optimization/traj_opt*.so "$PCT_COPY_TARGET/"
cp "$PCT_NATIVE_BUILD"/src/ele_planner/ele_planner*.so "$PCT_COPY_TARGET/"
cp "$PCT_NATIVE_BUILD"/src/map_manager/py_map_manager*.so "$PCT_COPY_TARGET/"
```

上述 tar 增量覆盖修改和新增文件，但保留 OSQP 已配置头，也不会移除 source-view 中
已从仓库删除或改名的旧文件。若确实修改 OSQP 配置，应重新执行第 2.5 节，而不是用
仓库模板替换该头。发生删除/改名或其他 third-party 变化时，停止并由用户选择归档旧
build 根或建立新版本化 build 根，再做完整构建；不要让 AI 擅自清理。

只修改 tomography 或 planner Python 文件时不需要重编 native modules。

## 5. 运行

需要发布 ROS topic 时先在一个 `enter-pct_planner` 终端运行 `roscore`。另一个同 profile
终端运行 tomography：

```bash
cd "$PCT_REPO/tomography/scripts"
python tomography.py \
  --scene whu_tls_forest \
  --pcd-file /home/workspace/simulation/data/path_planning/scene/WHU_TLS_Forest.pcd \
  --tomogram-file /home/workspace/simulation/data/AI_tests/pct/tomogram/WHU_TLS_Forest.pickle \
  --no-publish --no-spin
```

规划示例：

```bash
cd "$PCT_REPO/planner/scripts"
python plan.py \
  --scene whu_tls_forest --route 1 \
  --tomogram-file /home/workspace/simulation/data/AI_tests/pct/tomogram/WHU_TLS_Forest.pickle \
  --output-dir /home/workspace/simulation/data/AI_tests/pct/waypoint \
  --write-pcd --overwrite --no-spin
```

正式运行前应使用唯一输出目录；`--overwrite` 只能用于调用者已经确认允许替换的输出。

## 6. 构建产物与可删除性

| 路径 | 分类 | 能否删除 |
|---|---|---|
| `environment/build/.../gtsam-4.1.1/` | 依赖中间构建树 | 安装 config 和库验证后可重建；删除仍需用户授权 |
| `environment/build/.../osqp-0.6.2/` | 依赖中间构建树 | 安装 config 和库验证后可重建；删除仍需用户授权 |
| `environment/build/.../source-view/` | 隔离源码镜像 | 可重建，但保留的 native build 引用它；不能单独删除后继续增量编译 |
| `environment/build/.../native-modules/` | 构建树兼当前动态库来源 | 当前运行需要，不可按普通缓存删除；可通过完整重建恢复 |
| `environment/prefixes/.../native/` | GTSAM/OSQP 持久安装 | 不可删除；删后 profile 不能进入或模块不能链接 |
| `environment/prefixes/.../native-python/lib/` | 四个 runtime Python 扩展 | 不可删除；源码更新后需重新复制 |
| venv site-packages | Python/CUDA userspace 持久依赖 | 不可作为缓存删除 |
| tomogram、waypoint、PCD、验证 manifest | 项目数据或正式证据 | 不因任务完成自动删除；按数据索引和用户授权处理 |

每次任务报告必须列出构建及复制的文件，区分持久 runtime、中间构建、失败/拒绝产物
和正式算法输出，并明确哪些技术上可重建。可重建不代表已获删除授权。

## 7. 退出与故障处理

使用 `exit` 返回父 shell。ABI、CUDA、import 或动态链接失败时，保留失败命令、退出码、
完整 stderr、`CMakeCache.txt` 和 `ldd` 结果；不要通过追加 sibling GTSAM、Conda 或
`/usr/local` Python 路径掩盖身份错误。
