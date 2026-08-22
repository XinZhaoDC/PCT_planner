"""
广西森林环境数据
- 环境特点: 自然森林环境
- 难度: 高
- 应用: 森林监测和巡检

备选起终点坐标 (用于测试不同路径)
GX_Forest_GG2 场景的其他起终点组合:
- 路径1: [-57.879,51.645] -> [68.674,151.844]
- 路径2: [-34.043,139.025] -> [94.911,107.665]
- 路径3: [-14.333,1.933] -> [29.644,188.636]
- 路径4: [-41.898,149.540] -> [65.612,74.909] (当前使用)

这些坐标对应森林中不同的可通行区域，
可用于测试不同长度和复杂度的路径规划
"""

from pathlib import Path

from pct_config.model import RouteConfig, SceneConfig, SceneMap, SceneTrav


SCENE = SceneConfig(
    name="gx_forest_gg2",
    pcd_file=Path("rsc/pcd/GXForestGG2.pcd"),
    tomogram_file=Path("rsc/tomogram/GXForestGG2.pickle"),
    waypoint_dir=Path("rsc/waypoint"),
    waypoint_offset=(0.0, 0.0, 1.6),
    map=SceneMap(
        resolution=0.50,
        ground_h=0.0,  # 场景最低点
        slice_dh=1.0,
    ),
    trav=SceneTrav(
        kernel_size=0,
        interval_min=1.7,
        interval_free=1.7,
        slope_max=10.0,
        step_max=10.0,
        standable_ratio=0.001,
        cost_barrier=50.0,
        safe_margin=0.0,
        inflation=0.0,
    ),
    routes={
        "default": RouteConfig(
            start=(-41.898, 149.540),
            end=(65.612, 74.909),
            waypoint_file="GXForestGG2_waypoint",
            include_in_all=False,
        ),
        "1": RouteConfig(
            start=(-57.879, 51.645),
            end=(68.674, 151.844),
            waypoint_file="GXForestGG2_waypoint1",
        ),
        "2": RouteConfig(
            start=(-34.043, 139.025),
            end=(94.911, 107.665),
            waypoint_file="GXForestGG2_waypoint2",
        ),
        "3": RouteConfig(
            start=(-14.333, 1.933),
            end=(29.644, 188.636),
            waypoint_file="GXForestGG2_waypoint3",
        ),
        "4": RouteConfig(
            start=(-41.898, 149.540),
            end=(65.612, 74.909),
            waypoint_file="GXForestGG2_waypoint4",
        ),
    },
)
