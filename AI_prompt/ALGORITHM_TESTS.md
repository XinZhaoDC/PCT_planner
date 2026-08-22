# PCT Planner AI 算法测试记录

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
