# PCT Planner 重要更改记录

## 2026-10-09：WHU/Maan同步baseline回溯点

- annotated tag：`baseline-sim-whu-real-maan-20261009`。
- tagged commit：`195ba07ed807eecb595b516d6a3b3ae0d2f30ee7`，发布分支`main`。
- tag object：`8480506f447b1106d573a0734baca8aca028e6e1`。
- 组件职责：WHU场景规划、tomogram和waypoint导出。Maan为实采bag下游基线，不表示PCT新增了
  Maan仿真或已执行Maan规划测试。
- WHU关键配置：`pct_config/scenes/scene_whu_tls_forest.py`的导出Z offset为1.0；五条活动
  waypoint相对old WHU逐字节一致。
- 证据：`/home/workspace/simulation/.ai/results/HO-20261007-004-result.md`；
  发布记录：`/home/workspace/simulation/.ai/results/HO-20261009-001-result.md`。

同名tag在不同仓库指向不同commit；回溯时以tag剥离后的完整commit为准，不能把tag名称相同
解释为五个仓库具有相同SHA。本节是tag发布后的文档补充，已发布tag不移动；它不意味着
本节已包含在tag指向的历史tree中。验证证据冻结于对应任务，不用当前HEAD反向替换历史身份。

统一索引：[五仓库baseline索引](/home/workspace/Forest_Interface/AI_prompt/indexes/BASELINE_TAG_INDEX.md)。

## 2026-10-08：WHU waypoint 导出增量修正

- `scene_whu_tls_forest.py` 的 `waypoint_offset` 从 `(0.0, 0.0, 1.6)` 修正为 `(0.0, 0.0, 1.0)`。
- 该参数只改变 TXT/PCD 导出坐标（包括导出首末点），不改变规划器内部轨迹、优化、`RouteConfig.start/end` 或 ROS Path。
- 五条 WHU 路线使用 `z=1.0` 完整重生成后，与 `old_whu_waypoint/waypoint_1..5.pcd` 的点数、点序、XYZ、header 和整文件 SHA-256 分别完全一致；`z=1.6` 会让五条路线的全部导出点高约 `0.6 m`。
- README 的 Scene Configuration 已改为逐字段说明，并明确 offset 属于各 scene 的数据/消费坐标约定，不能把 dataclass fallback `1.6` 解释为所有场景的统一正确值。
- 完整复现证据：`/home/workspace/simulation/.ai/results/HO-20261007-004-result.md`。

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
