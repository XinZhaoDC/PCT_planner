"""
广西森林环境数据
- 环境特点: 自然森林环境
- 难度: 高
- 应用: 森林监测和巡检

备选起终点坐标 (用于测试不同路径)
GX_Forest_GG3 场景的其他起终点组合:
- 路径1: [-88.532,-67.000] -> [85.343,62.750]
- 路径2: [-18.594,-70.000] -> [20.844,70.500]
- 路径3: [71.532,-80.500] -> [-36.532,76.500]
- 路径4: [-86.218,-11.500] -> [77.656,0.500] (当前使用)

这些坐标对应森林中不同的可通行区域，
可用于测试不同长度和复杂度的路径规划
"""

from pathlib import Path

from pct_config.model import RouteConfig, SceneConfig, SceneMap, SceneTrav


SCENE = SceneConfig(
    name="gx_forest_gg3",
    pcd_file=Path("rsc/pcd/GXForestGG3.pcd"),
    tomogram_file=Path("rsc/tomogram/GXForestGG3.pickle"),
    waypoint_dir=Path("rsc/waypoint"),
    waypoint_offset=(0.0, 0.0, 1.6),
    map=SceneMap(
        resolution=0.50,
        ground_h=100.0,  # 场景最低点
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
            start=(-86.218, -11.500),
            end=(77.656, 0.500),
            waypoint_file="GXForestGG3_waypoint",
            include_in_all=False,
        ),
        "1": RouteConfig(
            start=(-88.532, -67.000),
            end=(85.343, 62.750),
            waypoint_file="GXForestGG3_waypoint1",
        ),
        "2": RouteConfig(
            start=(-18.594, -70.000),
            end=(20.844, 70.500),
            waypoint_file="GXForestGG3_waypoint2",
        ),
        "3": RouteConfig(
            start=(71.532, -80.500),
            end=(-36.532, 76.500),
            waypoint_file="GXForestGG3_waypoint3",
        ),
        "4": RouteConfig(
            start=(-86.218, -11.500),
            end=(77.656, 0.500),
            waypoint_file="GXForestGG3_waypoint4",
        ),
    },
)
