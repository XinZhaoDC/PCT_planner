"""
PCT 路径规划主程序

功能：
- 提供命令行接口运行路径规划任务
- 支持多种预定义场景的路径规划
- 集成 ROS 发布机制，实时发布规划结果
- 自动保存路径点文件供无人机使用

支持的场景：
- Spiral: 螺旋形障碍物场景
- Building: 建筑物环境
- Plaza: 广场开放环境
- WHUTLSForest: 武汉大学激光雷达森林数据
- ShenkanMLS: 沈勘移动激光扫描建筑数据
- GXForestGG2: 广西森林环境数据

使用方法：
python plan.py --scene Building
python plan.py --scene WHUTLSForest

输出：
- ROS 话题: /pct_path (nav_msgs/Path)
- 文件输出: /rsc/waypoint/newwaypoint.txt

依赖：
- ROS (rospy, nav_msgs)
- NumPy
- 自定义模块 (planner_wrapper, utils, config)
"""

import sys
import argparse
import numpy as np

import rospy
from nav_msgs.msg import Path

from utils import *
from planner_wrapper import TomogramPlanner

sys.path.append('../')
from config import Config

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default='Spiral', 
                   help='Name of the scene. Available: [\'Spiral\', \'Building\', \'Plaza\', \'WHUTLSForest\', \'ShenkanMLS\', \'GXForestGG2\']')
args = parser.parse_args()

# 加载配置
cfg = Config()

# 场景配置：根据场景名称设置对应的断层图文件和起终点
if args.scene == 'Spiral':
    """
    螺旋形障碍物场景
    - 环境特点: 螺旋形障碍物，需要绕行规划
    - 难度: 中等
    - 应用: 测试绕障能力
    """
    tomo_file = 'spiral0.3_2'
    start_pos = np.array([-16.0, -6.0], dtype=np.float32)
    end_pos = np.array([-26.0, -5.0], dtype=np.float32)
    
elif args.scene == 'Building':
    """
    建筑物环境场景
    - 环境特点: 复杂建筑结构，多层障碍物
    - 难度: 高
    - 应用: 城市环境导航
    """
    tomo_file = 'building2_9'
    start_pos = np.array([5.0, 5.0], dtype=np.float32)
    end_pos = np.array([-6.0, -1.0], dtype=np.float32)
    
elif args.scene == 'WHUTLSForest':
    """
    武汉大学激光雷达森林数据
    - 环境特点: 密集森林，树木障碍物
    - 数据来源: 地面激光雷达扫描
    - 难度: 高
    - 应用: 森林环境无人机导航
    """
    tomo_file = 'WHU_TLS_Forest'
    start_pos = np.array([13.40,19.27], dtype=np.float32)
    end_pos = np.array([43.05,60.35], dtype=np.float32)
    
elif args.scene == 'ShenkanMLS':
    """
    沈勘移动激光扫描建筑数据
    - 环境特点: 大型建筑群，复杂结构
    - 数据来源: 移动激光扫描系统
    - 难度: 极高
    - 应用: 大规模城市环境规划
    """
    tomo_file = 'shenkan_MLS_building'
    start_pos = np.array([54, 20], dtype=np.float32)
    end_pos = np.array([507, 115], dtype=np.float32)
    
elif args.scene == 'GXForestGG2':
    """
    广西森林环境数据
    - 环境特点: 自然森林环境
    - 难度: 高
    - 应用: 森林监测和巡检
    """
    tomo_file = 'GXForestGG2'
    start_pos = np.array([-41.898,149.540], dtype=np.float32)
    end_pos = np.array([65.612,74.909], dtype=np.float32)
    
else:
    """
    默认场景: 广场环境
    - 环境特点: 相对开阔，障碍物较少
    - 难度: 低
    - 应用: 基础测试和演示
    """
    tomo_file = 'plaza3_10'
    start_pos = np.array([0.0, 0.0], dtype=np.float32)
    end_pos = np.array([23.0, 10.0], dtype=np.float32)
    
# 备选起终点坐标 (用于测试不同路径)
'''
WHU-TLS Forest 场景的其他起终点组合:
- 路径1: [21.22,7.34] -> [14.93,62.67]
- 路径2: [35.96,6.53] -> [28.89,67.36]  
- 路径3: [45.33,16.55] -> [36.53,67.18]
- 路径4: [13.40,19.27] -> [43.05,60.35] (当前使用)
- 路径5: [9.19,26.52] -> [48.18,54.52]

这些坐标对应森林中不同的可通行区域，
可用于测试不同长度和复杂度的路径规划
'''

'''
Shenkan_MLS 场景的坐标说明:
- 完整3D坐标: [507.9080 115.0592 1.4566] -> [54.6010 20.4289 1.4659]
- 简化2D坐标: [54 20] -> [507 115] (当前使用)

注意: 坐标单位为米，基于激光雷达扫描的真实世界坐标系
'''

'''
GX_Forest_GG2 场景的其他起终点组合:
- 路径1: [-57.879,51.645] -> [68.674,151.844]
- 路径2: [-34.043,139.025] -> [94.911,107.665]  
- 路径3: [-14.333,1.933] -> [29.644,188.636]
- 路径4: [-41.898,149.540] -> [65.612,74.909] 


这些坐标对应森林中不同的可通行区域，
可用于测试不同长度和复杂度的路径规划
'''

# 输出配置
waypoint_file = tomo_file +'_waypoint'  # 路径点文件名 (不含扩展名)

# ROS 发布者设置
path_pub = rospy.Publisher("/pct_path", Path, latch=True, queue_size=1)

# 创建规划器实例
planner = TomogramPlanner(cfg)

def pct_plan():
    """
    执行路径规划的主函数
    
    流程：
    1. 加载指定场景的断层图数据
    2. 执行路径规划 (A* + 轨迹优化)
    3. 保存路径点文件
    4. 发布 ROS 路径消息
    5. 输出规划结果
    
    异常处理：
    - 如果规划失败 (traj_3d is None)，不执行后续操作
    - 规划成功则保存文件并发布消息
    """
    # 加载断层图数据
    planner.loadTomogram(tomo_file)

    # 执行路径规划
    traj_3d = planner.plan(start_pos, end_pos)
    
    if traj_3d is not None:
        # 保存路径点文件 (供无人机使用)
        planner.writeWaypoint(waypoint_file, traj_3d)
        
        # 发布 ROS 路径消息 (供 RViz 可视化)
        path_pub.publish(traj2ros(traj_3d))
        
        print("Trajectory published")
        print(f"Waypoints saved to: {waypoint_file}.txt")
        print(f"Trajectory points: {len(traj_3d)}")
    else:
        print("Path planning failed!")
        print("Possible reasons:")
        print("- No feasible path exists")
        print("- Start/end points are in obstacles")
        print("- Planning timeout")


if __name__ == '__main__':
    # 初始化 ROS 节点
    rospy.init_node("pct_planner", anonymous=True)

    # 执行路径规划
    pct_plan()

    # 保持节点运行 (用于ROS消息发布)
    rospy.spin()