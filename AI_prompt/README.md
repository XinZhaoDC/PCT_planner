# PCT Planner AI 辅助资料

## 分类入口

- [`IMPORTANT_CHANGES.md`](IMPORTANT_CHANGES.md)：场景配置、接口、依赖和复现方式的重要变化。
- [`ALGORITHM_TESTS.md`](ALGORITHM_TESTS.md)：AI 实际运行的算法测试、结果和分析。
- [`testing/`](testing/README.md)：检测脚本和规范运行资产。

## 固定位置环境、编译与运行

```bash
PCT_PREFIX=/home/workspace/simulation/environment/prefixes/pct_planner-noetic-py38-cu118
python3 -m venv "$PCT_PREFIX"
source "$PCT_PREFIX/bin/activate"
python -m pip install --no-index \
  --find-links=/home/workspace/simulation/download/pct_planner_py38 \
  numpy==1.24.4 cupy-cuda11x==12.3.0 fastrlock==0.8.3 \
  nvidia-cuda-runtime-cu11==11.8.89 nvidia-cuda-nvrtc-cu11==11.8.89 \
  nvidia-cublas-cu11==11.11.3.6 nvidia-cufft-cu11==10.9.0.58 \
  nvidia-curand-cu11==10.3.0.86 nvidia-cusolver-cu11==11.4.1.48 \
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

```bash
source /home/workspace/simulation/environment/shell_entries.sh
enter-pct_planner
roscore

# 另一个 enter-pct_planner 终端：
cd /home/workspace/simulation/code/pct_planner/src/PCT_planner/tomography/scripts
python tomography.py --scene whu_tls_forest \
  --pcd-file /home/workspace/simulation/data/path_planning/scene/WHU_TLS_Forest.pcd \
  --tomogram-file /home/workspace/simulation/data/AI_tests/pct/tomogram/WHU_TLS_Forest.pickle \
  --no-publish --no-spin

cd /home/workspace/simulation/code/pct_planner/src/PCT_planner/planner/scripts
python plan.py --scene whu_tls_forest --route 1 \
  --tomogram-file /home/workspace/simulation/data/AI_tests/pct/tomogram/WHU_TLS_Forest.pickle \
  --output-dir /home/workspace/simulation/data/AI_tests/pct/waypoint \
  --write-pcd --overwrite --no-spin
```

环境脚本正文不在仓库文档中复制。创建 profile 时保持入口只激活、不安装或构建。
