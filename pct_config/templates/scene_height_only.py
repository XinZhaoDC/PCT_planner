"""
仅使用高度相关惩罚的场景配置模板

特点：
- 禁用坡度惩罚（设置极大的坡度阈值）
- 禁用膨胀惩罚（设置膨胀半径为0）
- 只保留高度间隔相关的代价计算

复制到 pct_config/scenes/scene_<name>.py 后，必须补充实际路径和至少一条路线。
"""

from pathlib import Path

from pct_config.model import SceneConfig, SceneMap, SceneTrav


SCENE_TEMPLATE = SceneConfig(
    name="replace_me",
    pcd_file=Path("rsc/pcd/XXX.pcd"),  # 可以替换为你的点云文件
    tomogram_file=Path("rsc/tomogram/XXX.pickle"),
    waypoint_dir=Path("rsc/waypoint"),
    waypoint_offset=(0.0, 0.0, 1.6),
    map=SceneMap(
        resolution=0.10,
        ground_h=0.0,
        slice_dh=0.5,
    ),
    trav=SceneTrav(
        kernel_size=1,
        interval_min=0.50,  # 最小可通行间隔
        interval_free=0.65,  # 自由通行间隔阈值
        # 设置极大的坡度阈值，实际禁用坡度惩罚
        slope_max=10.0,  # 设置为极大值 (原来是 0.36)
        step_max=10.0,  # 设置为极大值 (原来是 0.20)
        standable_ratio=0.01,  # 设置为极小值，基本不考虑可站立区域
        cost_barrier=50.0,  # 障碍物代价
        # 禁用膨胀惩罚
        safe_margin=0.0,  # 缓冲区设为0
        inflation=0.0,  # 膨胀区设为0
    ),
    routes={},
)
