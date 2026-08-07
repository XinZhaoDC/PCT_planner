# PCT Planner AI 辅助资料

本目录只保存环境入口、验证方法和供后续 AI/维护者阅读的说明。算法源码、正式配置和运行时依赖仍保留在原来的职责目录中。

## 文件分类

- `environment/env_pct_planner.sh`：激活本容器 PCT 工作区运行环境，只设置 `PYTHONPATH`、`LD_LIBRARY_PATH` 和 `CUPY_CACHE_DIR`，不会安装或下载依赖。
- 算法入口仍是 `tomography/scripts/tomography.py` 和 `planner/scripts/plan.py`。
- 已安装的运行时文件仍在 `/home/workspace/code/pct_planner/.runtime`，不应移动进 Git 仓库。
- 可离线重装的 wheel 仍在 `/home/workspace/download/pct_planner_py38`。

## “工作区运行环境”的含义

`.runtime` 不是容器、虚拟机或 Conda 环境，而是放在工作区里的项目级安装目录。它包含 Python 3.8 site-packages、CUDA 11.8 用户态动态库、PCT 原生 Python 模块及其构建结果和 CuPy 缓存。环境脚本只是把这些已有文件加入当前 shell 的查找路径。

`download/pct_planner_py38` 保存的是依赖安装源/离线缓存，`.runtime` 保存的是已经安装并可直接加载的结果。迁移到 Ubuntu 20.04、Python 3.8、x86_64 且路径相同的容器时可以先复用 `.runtime` 做 import 检查；若 Python ABI、系统 ABI 或路径改变，应利用缓存 wheel 和源码重新生成 `.runtime`。

激活方式：

```bash
source /opt/ros/noetic/setup.bash
source /home/workspace/code/pct_planner/src/PCT_planner/AI_prompt/environment/env_pct_planner.sh
```

## WHU-TLS 验证结论

完整结果与量化报告位于：

```text
/home/workspace/data/pipeline_validation/WHU_TLS_Forest_20260806_145507
```

- 新生成 tomogram 与现有基线 SHA-256 完全相同。
- 五条 waypoint 的点数和 XY 坐标与现有结果完全相同。
- Z 不逐点相同：当前代码保留 `z += 1.6`，现有 Route 2-5 相当于 `z += 1.0`，所以新结果统一高 `0.6 m`；Route 1 另有 28 个历史手工高度修改点。

因此可以确认 PCT 的场景处理、路线选择和水平路径结果一致，但不能声称输出 waypoint 文件逐字节或 XYZ 全部一致。当前高度参数是有意保留的实验约定，比较时必须单独记录。
