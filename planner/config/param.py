"""
规划器配置参数模块

功能：
- 定义路径规划器的核心参数配置
- 场景路径、路线和导出偏移由 pct_config/scenes/scene_<name>.py 管理

配置结构：
- ConfigPlanner: 路径规划算法参数
- Config: 主配置类

使用方式：
from config import Config
cfg = Config()
planner = TomogramPlanner(cfg, waypoint_dir="/path/to/waypoint")

参数说明：
- use_quintic: 轨迹优化算法选择标志
- max_heading_rate: 航向角变化率限制
"""

class ConfigPlanner():
    """
    路径规划器算法参数配置
    
    属性：
    - use_quintic: 是否使用五次样条插值轨迹优化
        - True: 使用五次样条优化器 (无jerk约束)
        - False: 使用标准GPMP优化器
        - 影响: 轨迹平滑度和计算复杂度
    
    - max_heading_rate: 最大航向角变化率 (rad/s)
        - 默认值: 10 rad/s
        - 作用: 限制无人机转向速度，确保飞行安全
        - 影响: 轨迹的转弯半径和飞行时间
    """
    use_quintic = True          # 使用五次样条插值优化
    max_heading_rate = 10       # 最大航向角变化率 (rad/s)


class Config():
    """
    主配置类
    
    功能：
    - 提供统一的规划算法参数访问接口
    
    访问方式：
    cfg = Config()
    cfg.planner.use_quintic        # 访问规划器参数
    
    在代码中的使用：planner_wrapper.py 和 plan.py
    """
    planner = ConfigPlanner()   # 规划器算法参数
