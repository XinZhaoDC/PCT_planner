# PCT Planner AI 辅助资料

当前同步回溯点为`baseline-sim-whu-real-maan-20261009`。读取
[IMPORTANT_CHANGES.md](IMPORTANT_CHANGES.md)顶部的baseline发布记录，确认完整commit、依赖版本、
验证范围和有效配置；五仓库对应关系见
[统一baseline索引](/home/workspace/Forest_Interface/AI_prompt/indexes/BASELINE_TAG_INDEX.md)。

## 分类入口

- [`IMPORTANT_CHANGES.md`](IMPORTANT_CHANGES.md)：场景配置、接口、依赖和复现方式的重要变化。
- [`ALGORITHM_TESTS.md`](ALGORITHM_TESTS.md)：AI 实际运行的算法测试、结果和分析。
- [`testing/`](testing/README.md)：检测脚本和规范运行资产。

## 独立环境导航

固定 Python/CUDA/native 环境的首次建立、profile 入口、增量编译、运行和产物分类
统一见 [`INDEPENDENT_ENVIRONMENT.md`](../INDEPENDENT_ENVIRONMENT.md)。AI 不在本导航
中复制长命令或创建原生 CMake 不会读取的 defaults YAML；环境入口只激活，不安装
或构建。
