"""Data-only configuration objects shared by all PCT scenes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Tuple


Coordinate2D = Tuple[float, float]
Offset3D = Tuple[float, float, float]


@dataclass(frozen=True)
class SceneMap:
    resolution: float = 0.10  # 格网分辨率
    ground_h: float = 0.0  # 场景最低点
    slice_dh: float = 0.5  # 切片高度间隔 d_s


@dataclass(frozen=True)
class SceneTrav:
    kernel_size: int = 7  # 机器人占地格网大小
    interval_min: float = 0.50  # 最小机器人高度 d_min
    interval_free: float = 0.65  # 正常状态下的机器人高度 d_ref
    slope_max: float = 0.36  # 梯度阈值2 theta_s（不是直接使用的这个值）
    step_max: float = 0.20  # 梯度阈值1 theta_b（不是直接使用的这个值）
    standable_ratio: float = 0.20  # 可行走区域的占比 theta_p
    cost_barrier: float = 50.0  # 障碍物代价 C^B
    safe_margin: float = 0.4  # 缓冲区大小 d_sm
    inflation: float = 0.2  # 膨胀区大小 d_inf


@dataclass(frozen=True)
class RouteConfig:
    start: Coordinate2D
    end: Coordinate2D
    waypoint_file: str
    include_in_all: bool = True


@dataclass(frozen=True)
class SceneConfig:
    name: str
    map: SceneMap
    trav: SceneTrav
    pcd_file: Path
    tomogram_file: Path
    waypoint_dir: Path
    waypoint_offset: Offset3D = (0.0, 0.0, 1.6)
    routes: Mapping[str, RouteConfig] = field(default_factory=dict)
    default_route: str = "default"
