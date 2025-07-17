#!/usr/bin/env python3
"""
PCL GPU可视化器
===============

使用PCL和GPU加速分别显示LAS和PCD格式的点云数据，对比两者的显示效果。

功能：
1. 读取LAS文件并使用PCL显示
2. 读取PCD文件并使用PCL显示
3. 利用GPU加速渲染
4. 并排对比显示效果
5. 提供交互式查看功能

依赖：
- python-pcl
- laspy
- numpy
- Open3D (作为备选方案)
"""

import os
import sys
import time
import numpy as np
from typing import Optional, Tuple
import argparse

# 尝试导入PCL相关库
try:
    import pcl
    PCL_AVAILABLE = True
except ImportError:
    PCL_AVAILABLE = False
    print("警告: python-pcl 未安装，将使用 Open3D 作为替代方案")

# 尝试导入Open3D作为备选方案
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("警告: Open3D 未安装")

# 导入LAS处理库
try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    print("警告: laspy 未安装，无法读取LAS文件")


def read_las_file(las_file: str, max_points: int = 1000000) -> Optional[np.ndarray]:
    """
    读取LAS文件
    
    参数:
        las_file: LAS文件路径
        max_points: 最大点数（避免内存问题）
    
    返回:
        点云数据数组 (N, 3) 或 None
    """
    if not LASPY_AVAILABLE:
        print("错误: laspy 未安装，无法读取LAS文件")
        return None
    
    if not os.path.exists(las_file):
        print(f"错误: LAS文件不存在: {las_file}")
        return None
    
    try:
        print(f"读取LAS文件: {las_file}")
        
        # 读取LAS文件
        las = laspy.read(las_file)
        
        # 获取坐标
        x = las.x
        y = las.y
        z = las.z
        
        print(f"LAS文件总点数: {len(x)}")
        print(f"坐标范围:")
        print(f"  X: {x.min():.6f} - {x.max():.6f}")
        print(f"  Y: {y.min():.6f} - {y.max():.6f}")
        print(f"  Z: {z.min():.6f} - {z.max():.6f}")
        
        # 如果点数过多，进行采样
        if len(x) > max_points:
            print(f"点数过多，随机采样 {max_points} 个点")
            indices = np.random.choice(len(x), max_points, replace=False)
            x = x[indices]
            y = y[indices]
            z = z[indices]
        
        # 组合成点云数组
        points = np.column_stack((x, y, z))
        
        print(f"加载的点数: {len(points)}")
        
        return points
        
    except Exception as e:
        print(f"读取LAS文件时出错: {e}")
        return None


def read_pcd_file(pcd_file: str, max_points: int = 1000000) -> Optional[np.ndarray]:
    """
    读取PCD文件（支持ASCII和二进制格式）
    
    参数:
        pcd_file: PCD文件路径
        max_points: 最大点数（避免内存问题）
    
    返回:
        点云数据数组 (N, 3) 或 None
    """
    if not os.path.exists(pcd_file):
        print(f"错误: PCD文件不存在: {pcd_file}")
        return None
    
    try:
        print(f"读取PCD文件: {pcd_file}")
        
        # 优先使用Open3D读取PCD文件
        if OPEN3D_AVAILABLE:
            try:
                pcd = o3d.io.read_point_cloud(pcd_file)
                points = np.asarray(pcd.points)
                
                if len(points) == 0:
                    print("PCD文件中没有有效的点数据")
                    return None
                
                # 如果点数过多，进行采样
                if len(points) > max_points:
                    print(f"点数过多，随机采样 {max_points} 个点")
                    indices = np.random.choice(len(points), max_points, replace=False)
                    points = points[indices]
                
                print(f"PCD文件总点数: {len(points)}")
                print(f"坐标范围:")
                print(f"  X: {points[:, 0].min():.6f} - {points[:, 0].max():.6f}")
                print(f"  Y: {points[:, 1].min():.6f} - {points[:, 1].max():.6f}")
                print(f"  Z: {points[:, 2].min():.6f} - {points[:, 2].max():.6f}")
                
                return points
                
            except Exception as e:
                print(f"Open3D读取失败，尝试手动解析: {e}")
        
        # 回退到手动解析ASCII格式
        points = []
        point_count = 0
        
        # 尝试不同的编码格式
        for encoding in ['utf-8', 'latin-1', 'ascii']:
            try:
                with open(pcd_file, 'r', encoding=encoding) as f:
                    # 跳过头部
                    in_data = False
                    for line in f:
                        line = line.strip()
                        if line.startswith('DATA'):
                            in_data = True
                            continue
                        
                        if in_data and line:
                            try:
                                coords = [float(x) for x in line.split()]
                                if len(coords) >= 3:
                                    points.append(coords[:3])
                                    point_count += 1
                                    
                                    if point_count >= max_points:
                                        break
                            except ValueError:
                                continue
                    break
            except UnicodeDecodeError:
                continue
        
        if not points:
            print("PCD文件中没有有效的点数据")
            return None
        
        points = np.array(points)
        
        print(f"PCD文件总点数: {len(points)}")
        print(f"坐标范围:")
        print(f"  X: {points[:, 0].min():.6f} - {points[:, 0].max():.6f}")
        print(f"  Y: {points[:, 1].min():.6f} - {points[:, 1].max():.6f}")
        print(f"  Z: {points[:, 2].min():.6f} - {points[:, 2].max():.6f}")
        
        return points
        
    except Exception as e:
        print(f"读取PCD文件时出错: {e}")
        return None


def visualize_with_pcl(points: np.ndarray, title: str = "Point Cloud") -> bool:
    """
    使用PCL可视化点云
    
    参数:
        points: 点云数据 (N, 3)
        title: 窗口标题
    
    返回:
        是否成功
    """
    if not PCL_AVAILABLE:
        print("错误: PCL 不可用")
        return False
    
    try:
        print(f"使用PCL显示点云: {title}")
        
        # 创建PCL点云对象
        cloud = pcl.PointCloud(points.astype(np.float32))
        
        # 创建可视化器
        viewer = pcl.pcl_visualization.PCLVisualizer(title)
        
        # 设置背景色
        viewer.setBackgroundColor(0, 0, 0)
        
        # 添加点云
        viewer.addPointCloud(cloud, "cloud")
        
        # 设置点大小
        viewer.setPointCloudRenderingProperties(
            pcl.pcl_visualization.PCL_VISUALIZER_POINT_SIZE, 1, "cloud"
        )
        
        # 设置坐标轴
        viewer.addCoordinateSystem(1.0)
        
        # 启用GPU加速（如果支持）
        try:
            viewer.setUseVBO(True)  # 启用顶点缓冲对象
            print("已启用GPU加速")
        except:
            print("GPU加速不可用，使用CPU渲染")
        
        # 调整视角
        viewer.resetCameraViewpoint("cloud")
        
        print("点云显示窗口已打开，按 'q' 键退出")
        
        # 显示循环
        while not viewer.wasStopped():
            viewer.spinOnce(100)
            time.sleep(0.01)
        
        viewer.close()
        
        return True
        
    except Exception as e:
        print(f"PCL可视化时出错: {e}")
        return False


def visualize_with_open3d(points: np.ndarray, title: str = "Point Cloud") -> bool:
    """
    使用Open3D可视化点云（备选方案）
    
    参数:
        points: 点云数据 (N, 3)
        title: 窗口标题
    
    返回:
        是否成功
    """
    if not OPEN3D_AVAILABLE:
        print("错误: Open3D 不可用")
        return False
    
    try:
        print(f"使用Open3D显示点云: {title}")
        
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 估算法向量（用于更好的渲染效果）
        pcd.estimate_normals()
        
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=800, height=600)
        
        # 添加点云
        vis.add_geometry(pcd)
        
        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # 黑色背景
        opt.point_size = 1.0
        
        # 启用GPU加速
        try:
            opt.use_gpu = True
            print("已启用GPU加速")
        except:
            print("GPU加速不可用，使用CPU渲染")
        
        # 调整视角
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, 1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        
        print("点云显示窗口已打开，关闭窗口退出")
        
        # 显示循环
        vis.run()
        vis.destroy_window()
        
        return True
        
    except Exception as e:
        print(f"Open3D可视化时出错: {e}")
        return False


def compare_point_clouds(las_file: str, pcd_file: str, max_points: int = 1000000):
    """
    对比显示LAS和PCD点云
    
    参数:
        las_file: LAS文件路径
        pcd_file: PCD文件路径
        max_points: 最大点数
    """
    print("="*60)
    print("PCL GPU点云对比可视化器")
    print("="*60)
    
    # 读取LAS文件
    print("\n1. 读取LAS文件...")
    las_points = read_las_file(las_file, max_points)
    
    if las_points is None:
        print("无法读取LAS文件，跳过LAS显示")
    
    # 读取PCD文件
    print("\n2. 读取PCD文件...")
    pcd_points = read_pcd_file(pcd_file, max_points)
    
    if pcd_points is None:
        print("无法读取PCD文件，跳过PCD显示")
    
    # 数据对比
    if las_points is not None and pcd_points is not None:
        print("\n3. 数据对比:")
        print(f"LAS点数: {len(las_points)}")
        print(f"PCD点数: {len(pcd_points)}")
        
        # 坐标范围对比
        print("\nLAS坐标范围:")
        print(f"  X: {las_points[:, 0].min():.6f} - {las_points[:, 0].max():.6f}")
        print(f"  Y: {las_points[:, 1].min():.6f} - {las_points[:, 1].max():.6f}")
        print(f"  Z: {las_points[:, 2].min():.6f} - {las_points[:, 2].max():.6f}")
        
        print("\nPCD坐标范围:")
        print(f"  X: {pcd_points[:, 0].min():.6f} - {pcd_points[:, 0].max():.6f}")
        print(f"  Y: {pcd_points[:, 1].min():.6f} - {pcd_points[:, 1].max():.6f}")
        print(f"  Z: {pcd_points[:, 2].min():.6f} - {pcd_points[:, 2].max():.6f}")
        
        # 计算差异
        if len(las_points) == len(pcd_points):
            # 如果点数相同，计算坐标差异
            diff = np.abs(las_points - pcd_points)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"\n坐标差异:")
            print(f"  最大差异: {max_diff:.9f}")
            print(f"  平均差异: {mean_diff:.9f}")
            
            if max_diff < 1e-6:
                print("  结论: 坐标基本一致")
            else:
                print("  结论: 坐标存在差异")
    
    # 选择可视化库
    visualizer = None
    if PCL_AVAILABLE:
        visualizer = visualize_with_pcl
        print("\n4. 使用PCL进行可视化...")
    elif OPEN3D_AVAILABLE:
        visualizer = visualize_with_open3d
        print("\n4. 使用Open3D进行可视化...")
    else:
        print("\n错误: 没有可用的可视化库")
        return
    
    # 显示LAS点云
    if las_points is not None:
        print("\n显示LAS点云...")
        input("按回车键继续...")
        success = visualizer(las_points, f"LAS Point Cloud - {os.path.basename(las_file)}")
        if not success:
            print("LAS点云显示失败")
    
    # 显示PCD点云
    if pcd_points is not None:
        print("\n显示PCD点云...")
        input("按回车键继续...")
        success = visualizer(pcd_points, f"PCD Point Cloud - {os.path.basename(pcd_file)}")
        if not success:
            print("PCD点云显示失败")
    
    print("\n对比完成！")


def install_dependencies():
    """
    安装必要的依赖
    """
    print("安装依赖...")
    
    # 安装命令列表
    commands = [
        "pip install laspy",
        "pip install open3d",
        "pip install numpy",
        # PCL安装比较复杂，提供说明
    ]
    
    for cmd in commands:
        print(f"执行: {cmd}")
        os.system(cmd)
    
    print("\n注意: python-pcl 需要手动安装:")
    print("1. 安装PCL C++库:")
    print("   sudo apt-get install libpcl-dev")
    print("2. 安装python-pcl:")
    print("   pip install python-pcl")
    print("   或者从源码编译:")
    print("   git clone https://github.com/strawlab/python-pcl.git")
    print("   cd python-pcl")
    print("   python setup.py build_ext -i")
    print("   python setup.py install")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PCL GPU点云对比可视化器")
    parser.add_argument("--las", help="LAS文件路径")
    parser.add_argument("--pcd", help="PCD文件路径")
    parser.add_argument("--max-points", type=int, default=1000000, help="最大点数")
    parser.add_argument("--install-deps", action="store_true", help="安装依赖")
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dependencies()
        return 0
    
    if not args.las and not args.pcd:
        print("错误: 请提供LAS文件或PCD文件路径")
        print("使用 --help 查看帮助信息")
        return 1
    
    # 检查文件存在性
    if args.las and not os.path.exists(args.las):
        print(f"错误: LAS文件不存在: {args.las}")
        return 1
    
    if args.pcd and not os.path.exists(args.pcd):
        print(f"错误: PCD文件不存在: {args.pcd}")
        return 1
    
    # 检查库可用性
    if not PCL_AVAILABLE and not OPEN3D_AVAILABLE:
        print("错误: 没有可用的可视化库")
        print("请使用 --install-deps 安装依赖")
        return 1
    
    # 如果只提供了一个文件，单独显示
    if args.las and not args.pcd:
        print("只显示LAS文件...")
        las_points = read_las_file(args.las, args.max_points)
        if las_points is not None:
            if PCL_AVAILABLE:
                visualize_with_pcl(las_points, f"LAS - {os.path.basename(args.las)}")
            else:
                visualize_with_open3d(las_points, f"LAS - {os.path.basename(args.las)}")
    
    elif args.pcd and not args.las:
        print("只显示PCD文件...")
        pcd_points = read_pcd_file(args.pcd, args.max_points)
        if pcd_points is not None:
            if PCL_AVAILABLE:
                visualize_with_pcl(pcd_points, f"PCD - {os.path.basename(args.pcd)}")
            else:
                visualize_with_open3d(pcd_points, f"PCD - {os.path.basename(args.pcd)}")
    
    else:
        # 对比显示
        compare_point_clouds(args.las, args.pcd, args.max_points)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
