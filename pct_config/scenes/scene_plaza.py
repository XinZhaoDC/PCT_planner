"""
默认场景: 广场环境
- 环境特点: 相对开阔，障碍物较少
- 难度: 低
- 应用: 基础测试和演示
"""

from pathlib import Path

from pct_config.model import RouteConfig, SceneConfig, SceneMap, SceneTrav


SCENE = SceneConfig(
    name="plaza",
    pcd_file=Path("rsc/pcd/plaza3_10.pcd"),
    tomogram_file=Path("rsc/tomogram/plaza3_10.pickle"),
    waypoint_dir=Path("rsc/waypoint"),
    waypoint_offset=(0.0, 0.0, 1.6),
    map=SceneMap(
        resolution=0.10,
        ground_h=0.0,
        slice_dh=0.5,
    ),
    trav=SceneTrav(
        kernel_size=7,
        interval_min=0.50,
        interval_free=0.65,
        slope_max=0.36,
        step_max=0.17,
        standable_ratio=0.2,
        cost_barrier=50.0,
        safe_margin=0.4,
        inflation=0.2,
    ),
    routes={
        "default": RouteConfig(
            start=(0.0, 0.0),
            end=(23.0, 10.0),
            waypoint_file="plaza3_10_waypoint",
        ),
    },
)
