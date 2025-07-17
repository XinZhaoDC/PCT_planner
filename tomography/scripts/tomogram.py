"""
Tomogram 类 - 点云层析成像和可通行性分析

主要功能：
1. 点云数据处理和分层投影
2. 地形梯度计算
3. 可通行性代价评估
4. 代价膨胀和安全边界计算
5. 层简化和优化

该类是 PCT (Point Cloud Tomography) 系统的核心组件，
负责将3D点云数据转换为可用于路径规划的多层地图表示。
"""

import numpy as np
import cupy as cp

from kernels import *


class Tomogram(object):
    """
    点云层析成像处理类
    
    功能：
    - 处理3D点云数据，生成多层地图表示
    - 计算地形可通行性和安全代价
    - 提供GPU加速的高效计算
    """
    def __init__(self, cfg):
        """
        初始化 Tomogram 对象
        
        参数：
        - cfg: 配置对象，包含地图、可通行性等参数
        
        初始化的参数包括：
        - 地图分辨率和层间距
        - 可通行性评估参数
        - 安全边界和膨胀设置
        """
        # 地图基本参数
        self.resolution = cfg.map.resolution  # 网格分辨率 (米/像素)
        self.slice_dh = cfg.map.slice_dh      # 层间距 (米)
        
        # 可通行性分析参数
        self.half_trav_k_size = int(cfg.trav.kernel_size / 2)  # 遍历核半径
        self.interval_min = cfg.trav.interval_min              # 最小可通行间隔
        self.interval_free = cfg.trav.interval_free            # 自由通行间隔阈值
        
        # 地形坡度和步长限制
        self.step_stand = 1.2 * self.resolution * np.tan(cfg.trav.slope_max)  # 可站立的最大步长
        self.step_cross = cfg.trav.step_max                    # 可跨越的最大步长
        
        # 可站立网格阈值：基于邻域大小和可站立比例
        self.standable_th = int(cfg.trav.standable_ratio * (2 * self.half_trav_k_size + 1) ** 2) - 1
        
        # 代价和安全参数
        self.cost_barrier = float(cfg.trav.cost_barrier)       # 障碍物代价值
        self.safe_margin = cfg.trav.safe_margin                # 安全边界
        self.inflation = cfg.trav.inflation                    # 膨胀半径
        self.half_inf_k_size = int((self.safe_margin + self.inflation) / self.resolution)  # 膨胀核半径

    def initKernel(self):
        """
        初始化GPU内核函数
        
        功能：
        - 创建层析成像、可通行性分析和膨胀处理的GPU内核
        - 构建膨胀影响表，定义距离-影响权重关系
        
        内核包括：
        1. tomography_kernel: 点云分层投影
        2. trav_kernel: 可通行性代价计算
        3. inflation_kernel: 代价膨胀处理
        """
        # 创建层析成像内核
        self.tomography_kernel = tomographyKernel(
            self.resolution, 
            self.map_dim_x, 
            self.map_dim_y,
            self.n_slice_init,
            self.slice_h0,
            self.slice_dh
        )

        # 创建可通行性分析内核
        self.trav_kernel = travKernel(
            self.map_dim_x,
            self.map_dim_y,
            self.half_trav_k_size,
            self.interval_min,
            self.interval_free,
            self.step_cross, 
            self.step_stand, 
            self.standable_th, 
            self.cost_barrier
        )

        # 创建膨胀处理内核
        self.inflation_kernel = inflationKernel(
            self.map_dim_x,
            self.map_dim_y,
            self.half_inf_k_size
        )

        # 构建膨胀影响权重表
        self.inf_table = cp.zeros(
            (2 * self.half_inf_k_size + 1, 2 * self.half_inf_k_size + 1), 
            dtype=cp.float32
        )
        # 计算每个位置的距离权重
        for i in range(self.inf_table.shape[0]):
            for j in range(self.inf_table.shape[1]):
                # 计算到中心的欧几里得距离
                dist = np.sqrt(
                    (self.resolution * (i - self.half_inf_k_size)) ** 2 + \
                    (self.resolution * (j - self.half_inf_k_size)) ** 2
                )
                # 计算影响权重：距离越近权重越大
                self.inf_table[i, j] = np.clip(
                    1 - (dist - self.inflation) / (self.safe_margin + self.resolution),
                    a_min=0.0, a_max=1.0
                )

    def initBuffers(self):
        """
        初始化GPU内存缓冲区
        
        功能：
        - 为多层地图数据分配GPU内存
        - 创建用于存储不同处理阶段结果的缓冲区
        
        缓冲区包括：
        - layers_g: 地面高度层
        - layers_c: 天花板高度层
        - grad_mag_sq: 梯度幅值平方
        - grad_mag_max: 最大梯度幅值
        - trav_cost: 原始遍历代价
        - inflated_cost: 膨胀后的代价
        """
        self.layers_g = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.layers_c = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.grad_mag_sq = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.grad_mag_max = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.trav_cost = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.inflated_cost = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)

    def initMappingEnv(self, center, map_dim_x, map_dim_y, n_slice_init, slice_h0):
        """
        初始化地图环境参数
        
        参数：
        - center: 地图中心坐标 [x, y]
        - map_dim_x: 地图x方向维度（列数）
        - map_dim_y: 地图y方向维度（行数）
        - n_slice_init: 初始切片数量
        - slice_h0: 起始切片高度
        
        功能：
        - 设置地图的基本几何参数
        - 初始化内存缓冲区和GPU内核
        """
        self.center = cp.array(center, dtype=cp.float32)
        self.map_dim_x = int(map_dim_x)
        self.map_dim_y = int(map_dim_y)
        self.n_slice_init = int(n_slice_init)
        self.slice_h0 = float(slice_h0)
        
        self.initBuffers()
        self.initKernel()

    def clearMap(self):
        """
        清空地图数据
        
        功能：
        - 重置所有地图缓冲区为初始状态
        - 地面高度设为极小值 (-1e6)
        - 天花板高度设为极大值 (1e6)
        - 梯度和代价数据清零
        
        用途：
        - 在处理新的点云数据前重置地图状态
        """
        self.layers_g *= 0.
        self.layers_c *= 0.
        self.layers_g += -1e6  # 地面高度初始化为极小值
        self.layers_c += 1e6   # 天花板高度初始化为极大值

        self.grad_mag_sq *= 0.
        self.grad_mag_max *= 0.
        self.trav_cost *= 0.
        self.inflated_cost *= 0.

    def point2map(self, points):
        """
        将点云数据转换为多层地图表示
        
        参数：
        - points: 3D点云数据 (N x 3 数组: [x, y, z])
        
        返回：
        - layers_t: 膨胀后的遍历代价层
        - trav_gx: x方向遍历代价梯度
        - trav_gy: y方向遍历代价梯度
        - layers_g: 地面高度层
        - layers_c: 天花板高度层
        - t_gpu: GPU计算时间统计
        
        主要处理流程：
        1. 点云预处理和层析成像
        2. 梯度计算和可通行性分析
        3. 代价膨胀处理
        4. 层简化和优化
        """
        # 数据预处理：转换为GPU数组并移除NaN值
        points = cp.asarray(points)
        points = points[~cp.isnan(points).any(axis=1)]
        self.clearMap()

        # Tomogram
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        start_gpu.record()
        
        # 执行点云分层投影
        self.tomography_kernel(
            points, self.center, 
            self.layers_g, self.layers_c,
            size=(points.shape[0])
        )

        # 计算地面高度的梯度幅值
        # X方向梯度的平方（前向和后向差分的最大值）
        diff_x_sq = cp.maximum(
            (self.layers_g[:, 1:-1, :] - self.layers_g[:, :-2, :]) ** 2, 
            (self.layers_g[:, 1:-1, :] - self.layers_g[:,  2:, :]) ** 2
        )
        # Y方向梯度的平方
        diff_y_sq = cp.maximum(
            (self.layers_g[:, :, 1:-1] - self.layers_g[:, :, :-2]) ** 2, 
            (self.layers_g[:, :, 1:-1] - self.layers_g[:, :,  2:]) ** 2
        )
        # 总梯度幅值平方：两个方向的平方和
        self.grad_mag_sq[:, 1:-1, 1:-1] = diff_x_sq[:, :, 1:-1] + diff_y_sq[:, 1:-1, :]
        # 最大梯度幅值：两个方向的最大值
        self.grad_mag_max[:, 1:-1, 1:-1] = cp.maximum(diff_x_sq[:, :, 1:-1], diff_y_sq[:, 1:-1, :])
        
        # 计算垂直间隔（天花板高度 - 地面高度）
        interval = (self.layers_c - self.layers_g)

        end_gpu.record()
        end_gpu.synchronize()
        gpu_t_map = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

        # Traversability
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        start_gpu.record()

        # 计算基础遍历代价
        self.trav_kernel(
            interval, self.grad_mag_sq, self.grad_mag_max,
            self.trav_cost,
            size=(self.n_slice_init * self.map_dim_x * self.map_dim_y)
        )

        # 应用代价膨胀，扩大障碍物影响范围
        self.inflation_kernel(
            self.trav_cost, self.inf_table,
            self.inflated_cost,
            size=(self.n_slice_init * self.map_dim_x * self.map_dim_y)
        )

        end_gpu.record()
        end_gpu.synchronize()
        gpu_t_trav = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

        # === 阶段3: 层简化和优化 ===
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        start_gpu.record()

        # 层简化：移除冗余层，保留关键导航层
        idx_simp = [0]  # 始终保留第一层
        if self.layers_g.shape[0] > 1:
            l_idx, m_idx = 0, 1  # 下层索引和中层索引
            diff_h = self.layers_g[1:] - self.layers_g[:-1]  # 相邻层高度差
            
            while m_idx < self.n_slice_init - 2:
                # 检查层的唯一性和重要性
                mask_l_g = self.layers_g[m_idx] - self.layers_g[l_idx] > 0  # 地面高度有变化
                mask_l_t = self.inflated_cost[l_idx] > self.inflated_cost[m_idx]  # 代价有改善
                mask_u_g = diff_h[m_idx] > 0  # 与上层有高度差
                mask_t = self.inflated_cost[m_idx] < self.cost_barrier  # 可通行
                
                # 如果层具有独特性且可通行，则保留
                unique = (mask_l_g | mask_l_t) & mask_u_g & mask_t
                if cp.any(unique):
                    idx_simp.append(m_idx)
                    l_idx = m_idx
                m_idx += 1
            idx_simp.append(m_idx)  # 保留最后一层

        # 计算保留层的代价梯度（用于路径规划）
        trav_grad_x = (self.inflated_cost[idx_simp][:, 2:, :] - self.inflated_cost[idx_simp][:, :-2, :])
        trav_grad_y = (self.inflated_cost[idx_simp][:, :, 2:] - self.inflated_cost[idx_simp][:, :, :-2])
        
        end_gpu.record()
        end_gpu.synchronize()
        gpu_t_simp = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        
        # === 结果整理和返回 ===
        gpu_t_all = gpu_t_map + gpu_t_trav + gpu_t_simp
        #print("CuPy GPU time (ms):", gpu_t_all)

        # 将GPU数据转换为CPU数组，便于后续处理
        layers_t = self.inflated_cost[idx_simp].get()  # 最终的遍历代价层
        
        # 处理地面高度：将无效值(-1e6)替换为NaN
        layers_g = cp.where(
            self.layers_g[idx_simp] > -1e6, 
            self.layers_g[idx_simp],
            cp.nan
        ).get()
        
        # 处理天花板高度：将无效值(1e6)替换为NaN
        layers_c = cp.where(
            self.layers_c[idx_simp] < 1e6, 
            self.layers_c[idx_simp], 
            cp.nan
        ).get()
        
        # 构建完整的梯度数组（包含边界零值）
        trav_gx = np.zeros_like(layers_g)
        trav_gx[:, 1:-1, :] = trav_grad_x.get()
        trav_gy = np.zeros_like(layers_g)
        trav_gy[:, :, 1:-1] = trav_grad_y.get()

        # GPU计算时间统计
        t_gpu = {
            't_map': gpu_t_map,      # 层析成像时间
            't_trav': gpu_t_trav,    # 可通行性分析时间
            't_simp': gpu_t_simp,    # 层简化时间
        }
        
        return layers_t, trav_gx, trav_gy, layers_g, layers_c, t_gpu