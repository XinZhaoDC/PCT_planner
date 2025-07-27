#!/usr/bin/python3
"""
点云层析成像 ROS 节点

主要功能：
1. 加载点云数据 (PCD 文件)
2. 使用 GPU 加速进行点云层析成像处理
3. 生成多层地图表示
4. 计算可通行性和安全代价
5. 发布 ROS 点云消息用于可视化
6. 导出处理结果为 pickle 文件

该节点是 PCT (Point Cloud Tomography) 系统的主要入口点，
集成了数据加载、处理、可视化和导出功能。

使用方法：
python3 tomography.py --scene Plaza
"""

import os
import sys
import time
import pickle
import numpy as np
import open3d as o3d
  
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from tomogram import Tomogram

sys.path.append('../')
from config import POINT_FIELDS_XYZI, GRID_POINTS_XYZI
from config import Config

# 获取项目根目录路径
rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class Tomography(object):
    """
    点云层析成像处理类
    
    功能：
    - 管理点云数据的加载和预处理
    - 调用 GPU 加速的层析成像算法
    - 处理 ROS 消息发布和可视化
    - 导出处理结果为文件
    
    工作流程：
    1. 初始化配置和加载点云数据
    2. 执行层析成像处理
    3. 发布 ROS 消息用于可视化
    4. 导出结果文件
    """
    def __init__(self, cfg, scene_cfg):
        """
        初始化 Tomography 对象
        
        参数：
        - cfg: 全局配置对象 (包含 ROS 和导出设置)
        - scene_cfg: 场景配置对象 (包含点云文件和处理参数)
        
        初始化过程：
        1. 设置文件路径和基本参数
        2. 创建 Tomogram 处理对象
        3. 加载点云数据
        4. 执行层析成像处理
        """
        # 设置导出目录和点云文件路径
        self.export_dir = rsg_root + cfg.map.export_dir
        self.pcd_file = scene_cfg.pcd.file_name
        
        # 地图基本参数
        self.resolution = scene_cfg.map.resolution  # 网格分辨率
        self.ground_h = scene_cfg.map.ground_h      # 地面高度
        self.slice_dh = scene_cfg.map.slice_dh      # 切片间距

        # 初始化地图中心点
        self.center = np.zeros(2, dtype=np.float32)
        
        # 创建 Tomogram 处理对象
        self.tomogram = Tomogram(scene_cfg)
        
        # 加载点云数据
        points = self.loadPCD(self.pcd_file)

        # 执行层析成像处理
        self.process(points)

    def initROS(self):
        """
        初始化 ROS 发布者
        
        功能：
        - 创建点云数据发布者
        - 创建多层地图发布者 (地面层和天花板层)
        - 创建层析成像结果发布者
        
        发布的话题：
        - /global_points: 原始点云数据
        - /layer_G_*: 地面高度层 (按索引编号)
        - /layer_C_*: 天花板高度层 (按索引编号)
        - /tomogram: 综合层析成像结果
        """
        self.map_frame = cfg.ros.map_frame

        # 创建原始点云发布者
        pointcloud_topic = cfg.ros.pointcloud_topic
        self.pointcloud_pub = rospy.Publisher(pointcloud_topic, PointCloud2, latch=True, queue_size=1)

        # 创建地面层和天花板层发布者列表
        self.layer_G_pub_list = []
        self.layer_C_pub_list = []
        layer_G_topic = cfg.ros.layer_G_topic
        layer_C_topic = cfg.ros.layer_C_topic
        
        # 为每个切片层创建发布者
        for i in range(self.n_slice):
            layer_G_pub = rospy.Publisher(layer_G_topic + str(i), PointCloud2, latch=True, queue_size=1)
            self.layer_G_pub_list.append(layer_G_pub)
            layer_C_pub = rospy.Publisher(layer_C_topic + str(i), PointCloud2, latch=True, queue_size=1)
            self.layer_C_pub_list.append(layer_C_pub)

        # 创建综合层析成像结果发布者
        tomogram_topic = cfg.ros.tomogram_topic
        self.tomogram_pub = rospy.Publisher(tomogram_topic, PointCloud2, latch=True, queue_size=1)

    def loadPCD(self, pcd_file):
        """
        加载点云数据并计算地图参数
        
        参数：
        - pcd_file: 点云文件名 (相对于 rsc/pcd/ 目录)
        
        返回：
        - points: 处理后的点云数据 (N x 3 数组)
        
        功能：
        1. 使用 Open3D 加载 PCD 文件
        2. 计算点云边界和地图尺寸
        3. 确定地图中心点和切片参数
        4. 初始化 Tomogram 对象的映射环境
        5. 生成可视化原型数据
        """
        # 加载点云文件
        pcd = o3d.io.read_point_cloud(rsg_root + "/rsc/pcd/" + pcd_file)
        points = np.asarray(pcd.points).astype(np.float32)
        rospy.loginfo("PCD points: %d", points.shape[0])

        # 确保点云数据为 3D 坐标
        if points.shape[1] > 3:
            points = points[:, :3]
            
        # 计算点云边界
        self.points_max = np.max(points, axis=0)
        self.points_min = np.min(points, axis=0)           
        self.points_min[-1] = self.ground_h  # 设置地面高度
        
        # 计算地图尺寸 (网格数量)
        self.map_dim_x = int(np.ceil((self.points_max[0] - self.points_min[0]) / self.resolution)) + 4
        self.map_dim_y = int(np.ceil((self.points_max[1] - self.points_min[1]) / self.resolution)) + 4
        n_slice_init = int(np.ceil((self.points_max[2] - self.points_min[2]) / self.slice_dh))
        
        # 计算地图中心点
        self.center = (self.points_max[:2] + self.points_min[:2]) / 2
        self.slice_h0 = self.points_min[-1] + self.slice_dh
        
        # 初始化 Tomogram 对象的映射环境
        self.tomogram.initMappingEnv(self.center, self.map_dim_x, self.map_dim_y, n_slice_init, self.slice_h0)

        # 输出地图信息
        rospy.loginfo("Map center: [%.2f, %.2f]", self.center[0], self.center[1])
        rospy.loginfo("Dim_x: %d", self.map_dim_x)
        rospy.loginfo("Dim_y: %d", self.map_dim_y)
        rospy.loginfo("Num slices init: %d", n_slice_init)

        # 生成可视化网格原型
        self.VISPROTO_I, self.VISPROTO_P = \
            GRID_POINTS_XYZI(self.resolution, self.map_dim_x, self.map_dim_y)

        return points
        
    def process(self, points):
        """
        执行点云层析成像处理和性能基准测试
        
        参数：
        - points: 3D 点云数据
        
        功能：
        1. 多次运行层析成像算法进行性能基准测试
        2. 计算平均处理时间
        3. 导出处理结果
        4. 初始化 ROS 发布者并发布结果
        
        性能测试：
        - 运行 n_repeat + 1 次 (第一次为预热)
        - 使用 CUDA 事件进行精确时间测量
        - 分别统计各模块的处理时间
        """
        # 初始化时间统计变量
        t_map = 0.0    # 层析成像时间
        t_trav = 0.0   # 可通行性分析时间
        t_simp = 0.0   # 层简化时间
        t_all = 0.0    # 总时间
        n_repeat = 10  # 重复次数

        """
        GPU 时间基准测试
        
        使用 CUDA 事件同步进行精确时间测量
        重复运行 n_repeat 次计算平均处理时间
        排除第一次预热运行的时间，减少时间波动和初始调用开销
        详见: https://docs.cupy.dev/en/stable/user_guide/performance.html
        """
        for i in range(n_repeat + 1):
            t_start = time.time()
            
            # 执行层析成像处理
            layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c, t_gpu = self.tomogram.point2map(points)

            # 排除第一次预热运行，累积时间统计
            if i > 0:
                t_map += t_gpu['t_map']
                t_trav += t_gpu['t_trav']
                t_simp += t_gpu['t_simp']
                t_all += (time.time() - t_start) * 1e3

        # 输出性能统计信息
        rospy.loginfo("Num slices simp: %d", layers_g.shape[0])
        rospy.loginfo("Num repeats (for benchmarking only): %d", n_repeat)
        rospy.loginfo(" -- avg t_map  (ms): %f", t_map / n_repeat)
        rospy.loginfo(" -- avg t_trav (ms): %f", t_trav / n_repeat)
        rospy.loginfo(" -- avg t_simp (ms): %f", t_simp / n_repeat)
        rospy.loginfo(" -- avg t_all  (ms): %f", t_all / n_repeat)

        # 保存简化后的层数
        self.n_slice = layers_g.shape[0]

        # 导出处理结果
        map_file = os.path.splitext(self.pcd_file)[0]
        self.exportTomogram(np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c)), map_file)

        # 初始化 ROS 发布者并发布结果
        self.initROS()
        self.publishPoints(points)
        self.publishLayers(self.layer_G_pub_list, layers_g, layers_t)
        self.publishLayers(self.layer_C_pub_list, layers_c, None)
        self.publishTomogram(layers_g, layers_t)

    def exportTomogram(self, tomogram, map_file):
        """
        导出层析成像结果到 pickle 文件
        
        参数：
        - tomogram: 层析成像数据数组 (5 x n_slice x height x width)
                   包含: layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c
        - map_file: 输出文件名 (不含扩展名)
        
        功能：
        - 将处理结果保存为 pickle 文件
        - 包含数据、分辨率、中心点、切片参数等元信息
        - 使用 float16 格式减少文件大小
        """
        # 构建数据字典
        data_dict = {
            'data': tomogram.astype(np.float16),  # 转换为 float16 节省空间
            'resolution': self.resolution,        # 网格分辨率
            'center': self.center,                # 地图中心点
            'slice_h0': self.slice_h0,           # 起始切片高度
            'slice_dh': self.slice_dh,           # 切片间距
        }
        
        # 保存到 pickle 文件
        file_name = map_file + '.pickle'
        with open(self.export_dir + file_name, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        rospy.loginfo("Tomogram exported: %s", file_name)

    def publishPoints(self, points):
        """
        发布原始点云数据到 ROS 话题
        
        参数：
        - points: 3D 点云数据 (N x 3 数组)
        
        功能：
        - 将点云数据转换为 ROS PointCloud2 消息
        - 发布到 /global_points 话题用于可视化
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        # 创建点云消息并发布
        point_msg = pc2.create_cloud_xyz32(header, points)
        self.pointcloud_pub.publish(point_msg)

    def publishLayers(self, pub_list, layers, color=None):
        """
        发布多层地图数据到 ROS 话题
        
        参数：
        - pub_list: ROS 发布者列表
        - layers: 层数据 (n_slice x height x width)
        - color: 颜色数据 (可选，用于着色显示)
        
        功能：
        - 将每层数据转换为点云格式
        - 使用网格原型生成可视化点云
        - 为每层创建单独的 ROS 消息
        - 过滤掉无效点 (NaN 值)
        """
        header = Header()
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        # 复制可视化原型并调整到地图中心
        layer_points = self.VISPROTO_P.copy()
        layer_points[:, :2] += self.center

        # 为每层生成和发布点云
        for i in range(layers.shape[0]):
            # 设置高度值
            layer_points[:, 2] = layers[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            
            # 设置颜色/强度值
            if color is not None:
                layer_points[:, 3] = color[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            else:
                layer_points[:, 3] = 1.0
        
            # 过滤有效点并发布
            valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
            points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, valid_points)
            pub_list[i].publish(points_msg) 

    def publishTomogram(self, layers_g, layers_t):
        """
        发布综合层析成像结果到 ROS 话题
        
        参数：
        - layers_g: 地面高度层数据
        - layers_t: 遍历代价层数据
        
        功能：
        - 合并多层数据为单个点云
        - 消除相邻层间的重复点
        - 传播代价信息到相邻层
        - 创建综合可视化点云
        
        处理逻辑：
        1. 检查相邻层的高度差异
        2. 隐藏高度差异小的重复点
        3. 将较低代价传播到上层
        4. 合并所有有效点为单个点云
        """
        header = Header()
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        n_slice = layers_g.shape[0]
        
        # 复制数据用于可视化处理
        vis_g = layers_g.copy()
        vis_t = layers_t.copy() 
        layer_points = self.VISPROTO_P.copy()
        layer_points[:, :2] += self.center

        global_points = None
        
        # 处理除最后一层外的所有层
        for i in range(n_slice - 1):
            # 检查相邻层的高度差异
            mask_h = (vis_g[i + 1] - vis_g[i]) < self.slice_dh
            
            # 隐藏高度差异小的重复点
            vis_g[i, mask_h] = np.nan
            
            # 将较低代价传播到上层
            vis_t[i + 1, mask_h] = np.minimum(vis_t[i, mask_h], vis_t[i + 1, mask_h])
            
            # 生成当前层的点云
            layer_points[:, 2] = vis_g[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            layer_points[:, 3] = vis_t[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
            
            # 累积到全局点云
            if global_points is None:
                global_points = valid_points
            else:
                global_points = np.concatenate((global_points, valid_points), axis=0)

        # 处理最后一层
        layer_points[:, 2] = vis_g[-1, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
        layer_points[:, 3] = vis_t[-1, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
        valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
        global_points = np.concatenate((global_points, valid_points), axis=0)
        
        # 发布综合点云
        points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, global_points)
        self.tomogram_pub.publish(points_msg)


if __name__ == '__main__':
    """
    主程序入口点
    
    功能：
    1. 解析命令行参数
    2. 加载配置文件
    3. 初始化 ROS 节点
    4. 启动层析成像处理
    5. 保持节点运行
    
    支持的场景：
    - Spiral: 螺旋形场景
    - Building: 建筑物场景
    - Plaza: 广场场景
    - WHU_TLS_Forest: WHUTLS森林
    - ShenkanMLS: 深勘 MLS 建筑场景
    
    使用方法：
    python3 tomography.py --scene Plaza
    """
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='点云层析成像处理节点')
    parser.add_argument('--scene', type=str, required=True,
                       help='场景名称. 可选: [\'Spiral\', \'Building\', \'Plaza\', \'WHUTLSForest\', \'GXForestGG2\', \'ShenkanMLS\']')
    args = parser.parse_args()

    # 加载配置文件
    cfg = Config()  # 全局配置 (ROS 话题、导出路径等)
    scene_cfg = getattr(__import__('config'), 'Scene' + args.scene)  # 场景配置

    # 初始化 ROS 节点
    rospy.init_node('pointcloud_tomography', anonymous=True)

    # 创建并启动层析成像处理对象
    mapping = Tomography(cfg, scene_cfg)

    # 保持节点运行
    rospy.spin()