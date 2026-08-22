"""
建筑物环境场景
- 环境特点: 复杂建筑结构，多层障碍物
- 难度: 高
- 应用: 城市环境导航
"""

from pathlib import Path

from pct_config.model import RouteConfig, SceneConfig, SceneMap, SceneTrav


SCENE = SceneConfig(
    name="building",
    pcd_file=Path("rsc/pcd/building2_9.pcd"),
    tomogram_file=Path("rsc/tomogram/building2_9.pickle"),
    waypoint_dir=Path("rsc/waypoint"),
    waypoint_offset=(0.0, 0.0, 1.6),
    map=SceneMap(
        resolution=0.05,
        ground_h=0.0,
        slice_dh=0.5,
    ),
    trav=SceneTrav(
        kernel_size=7,
        interval_min=0.50,
        interval_free=0.65,
        slope_max=0.40,
        step_max=0.17,
        standable_ratio=0.20,
        cost_barrier=50.0,
        safe_margin=0.4,
        inflation=0.2,
    ),
    routes={
        "default": RouteConfig(
            start=(5.0, 5.0),
            end=(-6.0, -1.0),
            waypoint_file="building2_9_waypoint",
        ),
    },
)
