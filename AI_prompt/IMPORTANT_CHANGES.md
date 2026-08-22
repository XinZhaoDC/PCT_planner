# PCT Planner 重要更改记录

## 2026-08-21：共享场景配置重构

- 新增 `pct_config/`，`tomography.py` 和 `plan.py` 通过 loader 动态导入 `pct_config/scenes/scene_<name>.py`。
- 七个正式场景统一为小写 snake_case；每个场景模块集中保存 `SceneMap`、`SceneTrav`、输入输出路径、路线和 `waypoint_offset`。
- 删除主程序中的场景/WHU 路线分支；新增场景只需增加 `scene_<name>.py`，不维护注册表。
- WHU 五路线、GX 历史备选路线和 Shenkan 坐标说明保留；无 `--route` 时维持旧默认路线与文件名。
- `waypoint_offset` 只影响写出的 TXT/PCD，不改变优化器、ROS Path 或起终点。
- `scene_height_only.py` 移到模板目录，不参与正式场景发现。
- 重构没有调整 GTSAM、OSQP、native modules 或核心层析/规划算法。

## 环境兼容修正

历史隔离测试曾用 `sitecustomize.py` 追加 Ubuntu dist-packages，同时避免系统旧 NumPy 覆盖项目 NumPy。当前正式 profile 已直接组合 venv、native modules、planner、Noetic Python 和 Ubuntu dist-packages，不再需要该临时 shim。

后续重要变化继续按日期追加，保留 revision、文件、原因、验证和未验证边界。
