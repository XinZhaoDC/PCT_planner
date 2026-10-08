# PCT Planner AI 算法测试记录

## 2026-10-08：WHU 五路线导出增量复现

- 固定 PCT revision：`18ca40e0b713be8463ba184f90fc33b27370b293`；固定 tomogram SHA-256：`b7efbfbe59568009656fbcbb32f1146e9ccf9a2f48b3bcc5eb2c4bc1d1e3d507`。
- 候选 `z={0.0,1.0,1.6}`；`z=0.0` 五条均约低 `1.0 m`，`z=1.6` 五条均约高 `0.6 m`。
- `z=1.0` 通过当前正式规划入口逐路线完整生成，点数为 496、541、451、379、352；五条路线相对旧基线的逐点 `dx/dy/dz`、3D max 和 RMSE 全为 `0`，五个 PCD 整文件 SHA-256 也分别一致。
- 当前活动 Route 1 相对旧 Route 1 的 28 个非恒定 Z 差来自历史人工编辑，不属于 planner/export offset。
- 证据：`/home/workspace/simulation/AI_test/HO-20261007-004/` 和 `/home/workspace/simulation/.ai/results/HO-20261007-004-result.md`。

## 2026-08-21：WHU-TLS 全点云与五路线回归

- 环境：`enter-pct_planner`。
- 输入：15,095,947 点，SHA-256 `79cd01072bd2848376d9ab4fee3b0a372af538f8f14d16070add46ac5f0749af`。
- tomogram SHA-256：`b7efbfbe59568009656fbcbb32f1146e9ccf9a2f48b3bcc5eb2c4bc1d1e3d507`，与当前代码基线一致。
- 五条路线点数：496、541、451、379、352；TXT 相对当前代码验证基线的最大 XYZ 误差均为 `0.0`。
- 同批 TXT 与二进制 PCD 最大误差不超过 `3.8e-6 m`，来自 float32 写出。
- 相对更早 waypoint，五路线点数和 XY 一致；路线 2-5 保留 `+0.6 m` Z 差异，路线 1 有 28 个历史手工高度点，不是本次重构回归。
- 配置单元测试 7 项和 Python 编译检查通过；后者属于静态验证，不替代上述真实算法运行。
- 量化结果：`/home/workspace/simulation/data/AI_tests/pct-v2-20260821-090418/waypoint_comparison.json`。

后续测试必须记录输入、profile、命令、唯一输出、指标、失败和分析，不覆盖本结果。
