"""
Utils 包初始化文件

功能：
- 定义包的公共 API
- 将子模块中的函数提升到包级别
- 简化导入路径

使用方式：
from utils import transTrajGrid2Map, traj2ros
或
from utils import *

导入的函数：
- transTrajGrid2Map: 轨迹坐标转换函数 (来自 convertion.py)
- traj2ros: 轨迹 ROS 可视化函数 (来自 vis_ros.py)
"""

# 导入坐标转换相关函数
from .convertion import transTrajGrid2Map  # 将网格坐标轨迹转换为地图坐标

# 导入 ROS 可视化相关函数  
from .vis_ros import traj2ros              # 将轨迹转换为 ROS 消息格式

