"""
螺旋形障碍物场景
- 环境特点: 螺旋形障碍物，需要绕行规划
- 难度: 中等
- 应用: 测试绕障能力
"""

from pathlib import Path

from pct_config.model import RouteConfig, SceneConfig, SceneMap, SceneTrav


SCENE = SceneConfig(
    name="spiral",
    pcd_file=Path("rsc/pcd/spiral0.3_2.pcd"),
    tomogram_file=Path("rsc/tomogram/spiral0.3_2.pickle"),
    waypoint_dir=Path("rsc/waypoint"),
    waypoint_offset=(0.0, 0.0, 1.6),
    map=SceneMap(
        resolution=0.20,
        ground_h=0.0,
        slice_dh=0.5,
    ),
    trav=SceneTrav(
        kernel_size=7,
        interval_min=0.50,
        interval_free=0.65,
        slope_max=0.40,
        step_max=0.30,
        standable_ratio=0.40,
        cost_barrier=50.0,
        safe_margin=1.2,
        inflation=0.2,
    ),
    routes={
        "default": RouteConfig(
            start=(-16.0, -6.0),
            end=(-26.0, -5.0),
            waypoint_file="spiral0.3_2_waypoint",
        ),
    },
)
