"""
路径规划器包装模块

功能：
- 提供基于断层图 (Tomogram) 的路径规划功能
- 集成 A* 路径搜索和轨迹优化算法
- 支持多层地图的 3D 路径规划
- 提供路径点文件输出功能

主要组件：
- TomogramPlanner: 核心规划器类，封装了完整的规划流程
- 地图加载: 从 pickle 文件加载断层图数据
- 路径搜索: 使用 A* 算法进行路径搜索
- 轨迹优化: 使用 GPMP 优化器生成平滑轨迹
- 坐标转换: 处理网格坐标与世界坐标的转换

数据流：
1. 加载断层图数据 (.pickle 文件)
2. 初始化规划器和地图
3. 接收起点和终点坐标
4. 执行路径搜索和优化
5. 生成 3D 轨迹
6. 输出路径点文件

依赖关系：
- lib.a_star: A* 路径搜索算法
- lib.ele_planner: 高程感知规划器
- lib.traj_opt: 轨迹优化器
- utils: 坐标转换工具函数
"""

import os
import sys
import pickle
import numpy as np

from utils import *

sys.path.append('../')
from lib import a_star, ele_planner, traj_opt

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class TomogramPlanner(object):
    """
    基于断层图的路径规划器
    
    功能：
    - 加载和解析断层图数据
    - 执行多层地图的路径规划
    - 生成平滑的 3D 轨迹
    - 输出路径点文件
    
    属性：
    - cfg: 配置对象，包含规划器参数
    - resolution: 地图分辨率 (m/pixel)
    - center: 地图中心点坐标
    - n_slice: 层数 (高度层)
    - slice_h0: 第一层的高度
    - slice_dh: 层间高度差
    - map_dim: 地图维度 [width, height]
    - offset: 地图中心偏移
    """
    
    def __init__(self, cfg):
        """
        初始化规划器
        
        参数：
        - cfg: 配置对象，包含以下字段：
            - planner.use_quintic: 是否使用五次样条插值
            - planner.max_heading_rate: 最大航向角变化率
            - wrapper.tomo_dir: 断层图文件目录
            - wrapper.waypoint_dir: 路径点输出目录
        """
        self.cfg = cfg

        # 规划器参数
        self.use_quintic = self.cfg.planner.use_quintic
        self.max_heading_rate = self.cfg.planner.max_heading_rate

        # 文件路径
        self.tomo_dir = rsg_root + self.cfg.wrapper.tomo_dir
        self.waypoint_dir = rsg_root + self.cfg.wrapper.waypoint_dir

        # 地图参数 (将在加载断层图时初始化)
        self.resolution = None        # 地图分辨率
        self.center = None           # 地图中心点
        self.n_slice = None          # 层数
        self.slice_h0 = None         # 第一层高度
        self.slice_dh = None         # 层间高度差
        self.map_dim = []            # 地图维度 [width, height]
        self.offset = None           # 中心偏移

        # 起点和终点索引 (网格坐标)
        self.start_idx = np.zeros(3, dtype=np.int32)
        self.end_idx = np.zeros(3, dtype=np.int32)

    def loadTomogram(self, tomo_file):
        """
        加载断层图数据
        
        参数：
        - tomo_file: 断层图文件名 (不含扩展名)
        
        功能：
        1. 从 pickle 文件加载断层图数据
        2. 解析地图参数和几何信息
        3. 提取各层数据 (可通行性、梯度、高程等)
        4. 初始化规划器
        
        断层图数据结构：
        - data: 5层数据 [可通行性, 梯度X, 梯度Y, 地面高程, 顶棚高程]
        - resolution: 地图分辨率
        - center: 地图中心坐标
        - slice_h0: 第一层高度
        - slice_dh: 层间高度差
        """
        with open(self.tomo_dir + tomo_file + '.pickle', 'rb') as handle:
            data_dict = pickle.load(handle)

            # 加载断层图数据 (5层 x 高度层 x 宽度 x 高度)
            tomogram = np.asarray(data_dict['data'], dtype=np.float32)

            # 解析地图参数
            self.resolution = float(data_dict['resolution'])
            self.center = np.asarray(data_dict['center'], dtype=np.double)
            self.n_slice = tomogram.shape[1]  # 高度层数
            self.slice_h0 = float(data_dict['slice_h0'])
            self.slice_dh = float(data_dict['slice_dh'])
            self.map_dim = [tomogram.shape[2], tomogram.shape[3]]
            self.offset = np.array([int(self.map_dim[0] / 2), int(self.map_dim[1] / 2)], dtype=np.int32)

        # 提取各层数据
        trav = tomogram[0]      # 可通行性代价
        trav_gx = tomogram[1]   # X 方向梯度
        trav_gy = tomogram[2]   # Y 方向梯度
        elev_g = tomogram[3]    # 地面高程
        elev_g = np.nan_to_num(elev_g, nan=-100)  # 将 NaN 替换为 -100
        elev_c = tomogram[4]    # 顶棚高程
        elev_c = np.nan_to_num(elev_c, nan=1e6)   # 将 NaN 替换为很大的值

        # 初始化规划器
        self.initPlanner(trav, trav_gx, trav_gy, elev_g, elev_c)
        
    def initPlanner(self, trav, trav_gx, trav_gy, elev_g, elev_c):
        """
        初始化规划器和地图数据
        
        参数：
        - trav: 可通行性代价图
        - trav_gx: X 方向梯度
        - trav_gy: Y 方向梯度  
        - elev_g: 地面高程
        - elev_c: 顶棚高程
        
        功能：
        1. 计算层间可通行性变化
        2. 识别上升/下降通道 (gateway)
        3. 初始化 C++ 规划器模块
        """
        # 计算层间可通行性差异
        diff_t = trav[1:] - trav[:-1]
        diff_g = np.abs(elev_g[1:] - elev_g[:-1])

        # 识别上升通道
        gateway_up = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t < -8.0  # 可通行性显著改善
        mask_g = (diff_g < 0.1) & (~np.isnan(elev_g[1:]))  # 高程变化小
        gateway_up[:-1] = np.logical_and(mask_t, mask_g)

        # 识别下降通道
        gateway_dn = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t > 8.0   # 可通行性显著恶化
        mask_g = (diff_g < 0.1) & (~np.isnan(elev_g[:-1]))  # 高程变化小
        gateway_dn[1:] = np.logical_and(mask_t, mask_g)
        
        # 生成通道标记
        gateway = np.zeros_like(trav, dtype=np.int32)
        gateway[gateway_up] = 2    # 上升通道
        gateway[gateway_dn] = -2   # 下降通道

        # 初始化 C++ 规划器
        self.planner = ele_planner.OfflineElePlanner(
            max_heading_rate=self.max_heading_rate, use_quintic=self.use_quintic
        )
        
        # 初始化地图数据
        self.planner.init_map(
            20, 15,                    # 规划器内部参数
            self.resolution,           # 地图分辨率
            self.n_slice,             # 层数
            0.2,                      # 层间连接权重
            trav.reshape(-1, trav.shape[-1]).astype(np.double),
            elev_g.reshape(-1, elev_g.shape[-1]).astype(np.double),
            elev_c.reshape(-1, elev_c.shape[-1]).astype(np.double),
            gateway.reshape(-1, gateway.shape[-1]),
            trav_gy.reshape(-1, trav_gy.shape[-1]).astype(np.double),
            -trav_gx.reshape(-1, trav_gx.shape[-1]).astype(np.double)  # 注意负号
        )

    def plan(self, start_pos, end_pos):
        # TODO: calculate slice index. By default the start and end pos are all at slice 0
        """
        执行路径规划
        
        参数：
        - start_pos: 起点坐标 [x, y, z]
        - end_pos: 终点坐标 [x, y, z]
        
        返回：
        - traj_3d: 3D 轨迹 (N x 3 数组) 或 None (规划失败)
        
        规划流程：
        1. 将世界坐标转换为网格索引
        2. 执行 A* 路径搜索
        3. 进行轨迹优化 (GPMP)
        4. 转换回世界坐标
        """
        # 坐标转换：世界坐标 -> 网格索引
        # TODO: 计算层索引，目前默认起点和终点都在第 0 层
        self.start_idx[1:] = self.pos2idx(start_pos)
        self.end_idx[1:] = self.pos2idx(end_pos)

        # 执行路径规划
        self.planner.plan(self.start_idx, self.end_idx, True)
        
        # 获取 A* 路径搜索结果
        path_finder: a_star.Astar = self.planner.get_path_finder()
        path = path_finder.get_result_matrix()
        if len(path) == 0:
            return None  # 规划失败

        # 获取轨迹优化器
        optimizer: traj_opt.GPMPOptimizer = (
            self.planner.get_trajectory_optimizer()
            if not self.use_quintic
            else self.planner.get_trajectory_optimizer_wnoj()
        )

        # 获取优化结果
        opt_init = optimizer.get_opt_init_value()    # 初始轨迹
        init_layer = optimizer.get_opt_init_layer()  # 初始层索引
        traj_raw = optimizer.get_result_matrix()     # 优化后轨迹 (2D)
        layers = optimizer.get_layers()              # 层索引
        heights = optimizer.get_heights()            # 高度值

        # 组合轨迹数据
        opt_init = np.concatenate([opt_init.transpose(1, 0), init_layer.reshape(-1, 1)], axis=-1)
        traj = np.concatenate([traj_raw, layers.reshape(-1, 1)], axis=-1)
        
        # 构建 3D 轨迹
        y_idx = (traj.shape[-1] - 1) // 2
        traj_3d = np.stack([traj[:, 0], traj[:, y_idx], heights / self.resolution], axis=1)
        
        # 转换回世界坐标
        traj_3d = transTrajGrid2Map(self.map_dim, self.center, self.resolution, traj_3d)

        return traj_3d
    
    def pos2idx(self, pos):
        """
        将世界坐标转换为网格索引
        
        参数：
        - pos: 世界坐标 [x, y, z]
        
        返回：
        - idx: 网格索引 [col, row] (注意坐标轴交换)
        """
        pos = pos - self.center  # 相对于中心的坐标
        idx = np.round(pos / self.resolution).astype(np.int32) + self.offset
        idx = np.array([idx[1], idx[0]], dtype=np.float32)  # 交换 x, y 坐标
        return idx
    
    def writeWaypoint(self, waypoint_file, traj):
        """
        将轨迹写入路径点文件
        
        参数：
        - waypoint_file: 输出文件名 (不含扩展名)
        - traj: 3D 轨迹 (N x 3 数组)
        
        功能：
        1. 将轨迹点写入文本文件
        2. 格式：每行一个点 "x,y,z"
        3. 自动添加高度偏移 (z + 1.6)
        
        注意：
        - 使用追加模式 ('a+')
        - 高度偏移用于补偿无人机离地高度
        """
        with open(self.waypoint_dir + waypoint_file + '.txt', 'a+') as f:
            for waypoint in traj:
                x = waypoint[0]
                y = waypoint[1]
                z = waypoint[2]
                # Z 值计算说明：
                # 1. z 来自 optimizer.get_heights()，这是经过高度平滑的结果
                # 2. get_heights() 返回的高度计算过程：
                #    - 基础高度 = map_->GetHeight(layer, x, y) + reference_height_
                #    - GetHeight() 从断层图的 elev_g (地面高程) 数据中获取
                #    - reference_height_ 默认为 0.1m (相对地面的飞行高度)
                #    - 然后通过 HeightSmoother 进行三次样条平滑
                # 3. 最终添加 1.6m 安全偏移，确保无人机保持足够离地高度
                # 
                # 总结：z = 地面高程 + 0.1m (reference_height_) + 平滑处理 + 1.6m (安全偏移)
                lines=[str(x) + ',', str(y) + ',', str(z + 1.6)+'\n']
                f.writelines(lines)
