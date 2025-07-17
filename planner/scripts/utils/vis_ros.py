"""
ROS 可视化工具模块

功能：
- 提供轨迹数据与 ROS 消息格式之间的转换
- 用于在 RViz 中可视化路径规划结果

主要函数：
- traj2ros: 将轨迹数据转换为 ROS Path 消息

ROS 消息类型：
- Path: 用于表示一系列连续的位姿点
- PoseStamped: 带时间戳的位姿信息
"""

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def traj2ros(traj):
    """
    将轨迹数据转换为 ROS Path 消息格式
    
    参数：
    - traj: 轨迹数据 (N x 3 或 N x 6 数组)
           格式: [[x1, y1, z1], [x2, y2, z2], ...]
           或: [[x1, y1, z1, roll1, pitch1, yaw1], ...]
    
    返回：
    - path_msg: ROS Path 消息，可直接发布到 ROS 话题
    
    功能说明：
    1. 创建 ROS Path 消息容器
    2. 为每个轨迹点创建 PoseStamped 消息
    3. 设置位置信息 (x, y, z)
    4. 设置默认方向 (朝向 w=1, 表示无旋转)
    5. 将所有位姿点添加到路径中
    
    使用场景：
    - 在 RViz 中可视化规划的路径
    - 发布路径给导航系统
    - 路径跟踪和监控
    """
    # 创建 ROS Path 消息
    path_msg = Path()
    path_msg.header.frame_id = "map"  # 设置坐标系为地图坐标系

    # 遍历轨迹中的每个路径点
    for waypoint in traj:
        # 创建带时间戳的位姿消息
        pose = PoseStamped()
        pose.header.frame_id = "map"  # 设置坐标系
        
        # 设置位置信息
        pose.pose.position.x = waypoint[0]  # X 坐标 (东方向)
        pose.pose.position.y = waypoint[1]  # Y 坐标 (北方向)  
        pose.pose.position.z = waypoint[2]  # Z 坐标 (高度)
        
        # 设置方向信息 (四元数表示)
        # w=1 表示无旋转，即朝向为默认方向
        pose.pose.orientation.w = 1
        # 其他方向分量默认为 0
        # pose.pose.orientation.x = 0
        # pose.pose.orientation.y = 0
        # pose.pose.orientation.z = 0
        
        # 将位姿点添加到路径中
        path_msg.poses.append(pose)

    return path_msg