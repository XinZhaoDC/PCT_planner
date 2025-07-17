"""
坐标转换工具模块

功能：
- 提供网格坐标系与地图坐标系之间的转换
- 用于路径规划结果的坐标变换

主要函数：
- transTrajGrid2Map: 将网格坐标轨迹转换为地图坐标轨迹
"""

import numpy as np


def transTrajGrid2Map(grid_dim, center, resolution, traj_grid):
    """
    将网格坐标系中的轨迹转换为地图坐标系中的轨迹
    
    参数：
    - grid_dim: 网格维度 [height, width] 或 [rows, cols]
    - center: 地图中心点坐标 [x, y] (地图坐标系)
    - resolution: 网格分辨率 (米/像素)
    - traj_grid: 网格坐标系中的轨迹 (N x 3 数组: [col, row, layer])
    
    返回：
    - traj_map: 地图坐标系中的轨迹 (N x 3 数组: [x, y, z])
    
    坐标系转换说明：
    1. 网格坐标系: 以网格左上角为原点，单位为像素
       - 原点: (0, 0)
       - X轴: 向右 (列方向)
       - Y轴: 向下 (行方向)
    
    2. 地图坐标系: 以地图中心为原点，单位为米
       - 原点: center
       - X轴: 向右 (东方向)
       - Y轴: 向上 (北方向)
    
    转换步骤：
    1. 将网格坐标原点从左上角移动到网格中心
    2. 从像素单位转换为米单位 (乘以 resolution)
    3. 从网格中心坐标转换为地图坐标 (加上 center)
    4. 交换 X 和 Y 轴以适应不同的坐标系约定
    """
    # 步骤1: 计算网格中心的偏移量
    # grid_dim[1] 是宽度(列数), grid_dim[0] 是高度(行数)
    offset = np.array([grid_dim[1] // 2, grid_dim[0] // 2, 0])
    
    # 步骤2: 设置地图中心坐标 (注意 x,y 顺序交换)
    # center[1] 对应 Y 坐标, center[0] 对应 X 坐标
    # 第三维设为 0.5 作为默认高度偏移
    center_ = np.array([center[1], center[0], 0.5])

    # 步骤3: 执行坐标转换
    # (traj_grid - offset): 将原点从左上角移动到网格中心
    # * resolution: 从像素单位转换为米单位
    # + center_: 从网格中心转换为地图坐标
    traj_grid = (traj_grid - offset) * resolution + center_

    # 步骤4: 重新排列坐标轴以匹配最终的坐标系约定
    # 将 [col, row, layer] 转换为 [x, y, z]
    # traj_grid[:, 1] -> x 坐标 (原来的行坐标)
    # traj_grid[:, 0] -> y 坐标 (原来的列坐标)  
    # traj_grid[:, 2] -> z 坐标 (层坐标)
    traj_map = np.stack([traj_grid[:, 1], traj_grid[:, 0], traj_grid[:, 2]], axis=1)

    return traj_map
