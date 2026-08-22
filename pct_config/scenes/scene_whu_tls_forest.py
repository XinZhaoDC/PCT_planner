"""
武汉大学激光雷达森林数据
- 环境特点: 密集森林，树木障碍物
- 数据来源: 地面激光雷达扫描
- 难度: 高
- 应用: 森林环境无人机导航

备选起终点坐标 (用于测试不同路径)
WHU-TLS Forest 场景的其他起终点组合:
- 路径1: [21.22,7.34] -> [14.93,62.67]
- 路径2: [35.96,6.53] -> [28.89,67.36]
- 路径3: [45.33,16.55] -> [36.53,67.18]
- 路径4: [13.40,19.27] -> [43.05,60.35] (当前使用)
- 路径5: [9.19,26.52] -> [48.18,54.52]

这些坐标对应森林中不同的可通行区域，
可用于测试不同长度和复杂度的路径规划
"""

from pathlib import Path

from pct_config.model import RouteConfig, SceneConfig, SceneMap, SceneTrav


SCENE = SceneConfig(
    name="whu_tls_forest",
    pcd_file=Path("/home/workspace/simulation/data/AI_tests/pct/WHU_TLS_Forest.pcd"),
    tomogram_file=Path("/home/workspace/simulation/data/AI_tests/pct/WHU_TLS_Forest.pickle"),
    waypoint_dir=Path("/home/workspace/simulation/data/AI_tests/pct"),
    waypoint_offset=(0.0, 0.0, 1.6),
    map=SceneMap(
        resolution=0.10,
        ground_h=5.0,  # 大于森林地面的最高处
        slice_dh=0.5,
    ),
    trav=SceneTrav(
        kernel_size=3,
        interval_min=1.6,
        interval_free=1.7,
        slope_max=1.0,
        step_max=1.0,
        standable_ratio=0.0,
        cost_barrier=50.0,
        safe_margin=0.0,
        inflation=0.05,
    ),
    routes={
        # Preserve the old output name when --route is omitted.
        "default": RouteConfig(
            start=(13.40, 19.27),
            end=(43.05, 60.35),
            waypoint_file="WHU_TLS_Forest_waypoint",
            include_in_all=False,
        ),
        "1": RouteConfig(
            start=(21.22, 7.34),
            end=(14.93, 62.67),
            waypoint_file="waypoint_1",
        ),
        "2": RouteConfig(
            start=(35.96, 6.53),
            end=(28.89, 67.36),
            waypoint_file="waypoint_2",
        ),
        "3": RouteConfig(
            start=(45.33, 16.55),
            end=(36.53, 67.18),
            waypoint_file="waypoint_3",
        ),
        "4": RouteConfig(
            start=(13.40, 19.27),
            end=(43.05, 60.35),
            waypoint_file="waypoint_4",
        ),
        "5": RouteConfig(
            start=(9.19, 26.52),
            end=(48.18, 54.52),
            waypoint_file="waypoint_5",
        ),
    },
)
