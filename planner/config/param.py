"""
规划器配置参数模块

功能：
- 定义路径规划器的核心参数配置
- 提供文件路径和算法参数的统一管理
- 支持参数的层次化组织和访问

配置结构：
- ConfigPlanner: 路径规划算法参数
- ConfigWrapper: 文件系统路径参数
- Config: 主配置类，整合所有子配置

使用方式：
from config import Config
cfg = Config()
planner = TomogramPlanner(cfg)

参数说明：
- use_quintic: 轨迹优化算法选择标志
- max_heading_rate: 航向角变化率限制
- tomo_dir: 断层图数据文件目录
- waypoint_dir: 路径点输出文件目录
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


class ConfigWrapper():
    """
    文件系统路径配置
    
    属性：
    - tomo_dir: 断层图数据文件目录
        - 相对路径: 相对于项目根目录
        - 存储内容: .pickle格式的断层图数据文件
        - 用途: 加载预处理的多层地图数据
    
    - waypoint_dir: 路径点输出文件目录
        - 相对路径: 相对于项目根目录
        - 存储内容: .txt格式的路径点文件
        - 用途: 保存规划结果供无人机使用
    """
    tomo_dir = '/rsc/tomogram/'     # 断层图数据目录
    waypoint_dir = '/rsc/waypoint/' # 路径点输出目录


class Config():
    """
    主配置类
    
    功能：
    - 整合所有子配置模块
    - 提供统一的配置访问接口
    - 支持层次化的参数组织
    
    访问方式：
    cfg = Config()
    cfg.planner.use_quintic        # 访问规划器参数
    cfg.wrapper.tomo_dir           # 访问路径参数
    
    在代码中的使用：
    1. planner_wrapper.py 中的 TomogramPlanner.__init__()
    2. plan.py 中的主程序入口
    """
    planner = ConfigPlanner()   # 规划器算法参数
    wrapper = ConfigWrapper()   # 文件系统路径参数