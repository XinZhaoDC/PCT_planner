"""
沈勘移动激光扫描建筑数据
- 环境特点: 大型建筑群，复杂结构
- 数据来源: 移动激光扫描系统
- 难度: 极高
- 应用: 大规模城市环境规划

Shenkan_MLS 场景的坐标说明:
- 完整3D坐标: [507.9080 115.0592 1.4566] -> [54.6010 20.4289 1.4659]
- 简化2D坐标: [54 20] -> [507 115] (当前使用)

注意: 坐标单位为米，基于激光雷达扫描的真实世界坐标系
"""

from pathlib import Path

from pct_config.model import RouteConfig, SceneConfig, SceneMap, SceneTrav


SCENE = SceneConfig(
    name="shenkan_mls",
    pcd_file=Path("rsc/pcd/shenkan_MLS_building.pcd"),
    tomogram_file=Path("rsc/tomogram/shenkan_MLS_building.pickle"),
    waypoint_dir=Path("rsc/waypoint"),
    waypoint_offset=(0.0, 0.0, 1.6),
    map=SceneMap(
        resolution=0.3,
        ground_h=1.0,
        slice_dh=1.0,
    ),
    trav=SceneTrav(
        kernel_size=1,
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
        "default": RouteConfig(
            start=(54.0, 20.0),
            end=(507.0, 115.0),
            waypoint_file="shenkan_MLS_building_waypoint",
        ),
    },
)
