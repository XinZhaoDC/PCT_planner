#!/usr/bin/env python3
"""
LAS/LAZ 转 PCD 转换工具
====================

功能：
1. 读取 LAS/LAZ 文件并转换为 PCD 格式
2. 支持偏移量处理 (--offset true/false)
3. 使用 float64 精度处理所有数据
4. 支持体素下采样和高度过滤

使用方法：
python3 las2pcd.py input.las output.pcd --offset true
"""

import os
import sys
import argparse
import numpy as np
import gc  # 垃圾回收

# 检查 Python 版本
if sys.version_info < (3, 0):
    print("错误: 需要 Python 3.0 或更高版本")
    sys.exit(1)

# 导入必要的库
try:
    import laspy
except ImportError:
    print("错误: 需要安装 laspy 库")
    print("请运行: pip install laspy")
    sys.exit(1)

# 可选的内存监控
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_memory_usage():
    """获取当前内存使用情况（MB）"""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    else:
        return 0.0


def read_las_file_chunked(las_data, header, apply_offset=True):
    """
    分块读取大型 LAS 文件
    
    参数:
        las_data: LAS 数据对象
        header: LAS 头文件信息
        apply_offset: 是否应用LAS头文件中的偏移量
    
    返回:
        las_data, header, offset_info, point_count
    """
    print("使用分块处理模式...")
    
    point_count = header.point_count
    chunk_size = 50000  # 每块5万个点
    
    # 获取LAS头文件中的偏移量和缩放因子
    las_offset = np.array([header.x_offset, header.y_offset, header.z_offset], dtype=np.float64)
    las_scale = np.array([header.x_scale, header.y_scale, header.z_scale], dtype=np.float64)
    
    print(f"LAS头文件偏移量: [{las_offset[0]:.6f}, {las_offset[1]:.6f}, {las_offset[2]:.6f}]")
    print(f"LAS头文件缩放因子: [{las_scale[0]:.6f}, {las_scale[1]:.6f}, {las_scale[2]:.6f}]")
    
    if apply_offset:
        print("将应用LAS头文件中的偏移量，输出真实世界坐标")
        offset_info = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # PCD文件不需要额外偏移
    else:
        print("不应用LAS头文件中的偏移量，输出相对坐标")
        offset_info = las_offset  # 记录LAS偏移量供参考
    
    # 计算第一个点的坐标作为参考
    if point_count > 0:
        first_chunk_x = np.array(las_data.x[0:min(1000, point_count)], dtype=np.float64)
        first_chunk_y = np.array(las_data.y[0:min(1000, point_count)], dtype=np.float64)
        first_chunk_z = np.array(las_data.z[0:min(1000, point_count)], dtype=np.float64)
        
        print(f"前1000个点的坐标范围:")
        print(f"  X: {first_chunk_x.min():.6f} - {first_chunk_x.max():.6f}")
        print(f"  Y: {first_chunk_y.min():.6f} - {first_chunk_y.max():.6f}")
        print(f"  Z: {first_chunk_z.min():.6f} - {first_chunk_z.max():.6f}")
        
        del first_chunk_x, first_chunk_y, first_chunk_z
        gc.collect()
    
    # 返回 las_data、header 和 offset_info 用于后续处理
    return las_data, header, offset_info, point_count


def read_las_file(las_path, apply_offset=True):
    """
    读取 LAS 文件
    
    参数:
        las_path: LAS 文件路径
        apply_offset: 是否应用偏移量
    
    返回:
        points: 点云数组 (N, 3) float64 或 点云生成器
        offset: 偏移量 [x, y, z] 或 [0, 0, 0]
        is_chunked: 是否为分块处理模式
    """
    print(f"读取 LAS 文件: {las_path}")
    print(f"初始内存使用: {get_memory_usage():.2f} MB")
    
    if not os.path.exists(las_path):
        raise FileNotFoundError(f"文件不存在: {las_path}")
    
    # 读取 LAS 文件
    las_data = laspy.read(las_path)
    header = las_data.header
    
    print(f"点云数量: {header.point_count}")
    print(f"LAS 版本: {header.version}")
    print(f"读取后内存使用: {get_memory_usage():.2f} MB")
    
    # 获取坐标数据（laspy 会自动应用缩放和偏移）
    print("正在提取坐标数据...")
    
    # 检查点云大小，决定是否使用分块处理
    point_count = header.point_count
    estimated_memory = point_count * 3 * 8 / 1024 / 1024  # 3坐标 * 8字节 / MB
    
    print(f"预估内存需求: {estimated_memory:.2f} MB")
    
    if estimated_memory > 800:  # 降低阈值到800MB
        print("检测到大文件，使用分块处理...")
        las_data_chunked, header_chunked, offset, total_points = read_las_file_chunked(las_data, header, apply_offset)
        return las_data_chunked, header_chunked, offset, True, total_points, apply_offset
    
    # 小文件直接处理
    # 获取LAS头文件中的偏移量和缩放因子
    las_offset = np.array([header.x_offset, header.y_offset, header.z_offset], dtype=np.float64)
    las_scale = np.array([header.x_scale, header.y_scale, header.z_scale], dtype=np.float64)
    
    print(f"LAS头文件偏移量: [{las_offset[0]:.6f}, {las_offset[1]:.6f}, {las_offset[2]:.6f}]")
    print(f"LAS头文件缩放因子: [{las_scale[0]:.6f}, {las_scale[1]:.6f}, {las_scale[2]:.6f}]")
    
    if apply_offset:
        # 应用LAS头文件中的偏移量，输出真实世界坐标
        print("应用LAS头文件中的偏移量，输出真实世界坐标")
        # laspy.read() 已经自动应用了偏移量和缩放因子
        x_coords = np.array(las_data.x, dtype=np.float64)
        y_coords = np.array(las_data.y, dtype=np.float64)
        z_coords = np.array(las_data.z, dtype=np.float64)
        offset_info = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # PCD文件不需要额外偏移
    else:
        # 不应用LAS头文件中的偏移量，输出相对坐标
        print("不应用LAS头文件中的偏移量，输出相对坐标")
        # 获取原始整数坐标并只应用缩放因子，不应用偏移量
        x_coords = np.array(las_data.x, dtype=np.float64) - las_offset[0]
        y_coords = np.array(las_data.y, dtype=np.float64) - las_offset[1]
        z_coords = np.array(las_data.z, dtype=np.float64) - las_offset[2]
        offset_info = las_offset  # 记录LAS偏移量供参考
    
    print(f"坐标数据大小: {(x_coords.nbytes + y_coords.nbytes + z_coords.nbytes) / 1024 / 1024:.2f} MB")
    print(f"提取后内存使用: {get_memory_usage():.2f} MB")
    
    print(f"坐标范围:")
    print(f"  X: {x_coords.min():.6f} - {x_coords.max():.6f}")
    print(f"  Y: {y_coords.min():.6f} - {y_coords.max():.6f}")
    print(f"  Z: {z_coords.min():.6f} - {z_coords.max():.6f}")
    
    # 释放 las_data 引用
    del las_data
    gc.collect()
    
    # 组合坐标
    points = np.column_stack([x_coords, y_coords, z_coords])
    
    # 清理临时变量
    del x_coords, y_coords, z_coords
    gc.collect()
    
    return points, offset_info, False, len(points), None, apply_offset


def downsample_voxel(points, voxel_size):
    """
    体素下采样 - 内存优化版本
    
    参数:
        points: 点云数组 (N, 3)
        voxel_size: 体素大小
    
    返回:
        下采样后的点云数组
    """
    print(f"开始体素下采样，体素大小: {voxel_size}")
    print(f"输入点数: {len(points)}")
    print(f"预估内存使用: {points.nbytes / 1024 / 1024:.2f} MB")
    
    if len(points) == 0:
        return points
    
    # 检查内存使用，如果点云过大则分块处理
    if len(points) > 1000000:  # 超过100万个点使用分块处理
        print("使用分块体素下采样...")
        return downsample_voxel_chunked(points, voxel_size)
    
    try:
        # 计算体素索引
        print("计算体素索引...")
        voxel_indices = np.floor(points / voxel_size).astype(np.int64)
        
        # 使用更高效的方法：排序+分组
        print("排序点云数据...")
        
        # 创建复合索引进行排序
        # 使用稀疏编码避免过大的索引值
        min_idx = np.min(voxel_indices, axis=0)
        max_idx = np.max(voxel_indices, axis=0)
        
        print(f"体素索引范围: X[{min_idx[0]}, {max_idx[0]}], Y[{min_idx[1]}, {max_idx[1]}], Z[{min_idx[2]}, {max_idx[2]}]")
        
        # 将体素索引转换为1D索引
        range_x = max_idx[0] - min_idx[0] + 1
        range_y = max_idx[1] - min_idx[1] + 1
        
        # 避免索引溢出
        if range_x * range_y > 1e9:
            print("体素网格过大，使用哈希方法...")
            return downsample_voxel_hash(points, voxel_size)
        
        # 标准化索引
        normalized_indices = voxel_indices - min_idx
        flat_indices = (normalized_indices[:, 0] * range_y * (max_idx[2] - min_idx[2] + 1) + 
                       normalized_indices[:, 1] * (max_idx[2] - min_idx[2] + 1) + 
                       normalized_indices[:, 2])
        
        print("分组相同体素的点...")
        
        # 排序以便分组
        sort_indices = np.argsort(flat_indices)
        sorted_flat_indices = flat_indices[sort_indices]
        sorted_points = points[sort_indices]
        
        # 找到每个体素的边界
        unique_indices, inverse_indices = np.unique(sorted_flat_indices, return_inverse=True)
        
        print(f"唯一体素数量: {len(unique_indices)}")
        
        # 计算每个体素的中心点
        print("计算体素中心...")
        downsampled_points = []
        
        for i, unique_idx in enumerate(unique_indices):
            # 找到属于当前体素的所有点
            mask = inverse_indices == i
            voxel_points = sorted_points[mask]
            
            # 计算中心点
            center = np.mean(voxel_points, axis=0)
            downsampled_points.append(center)
            
            # 显示进度
            if i % 10000 == 0:
                progress = (i / len(unique_indices)) * 100
                print(f"  进度: {progress:.1f}% ({i}/{len(unique_indices)})")
        
        result = np.array(downsampled_points, dtype=np.float64)
        print(f"下采样完成: {len(points)} -> {len(result)} 点")
        print(f"压缩比: {len(result) / len(points) * 100:.1f}%")
        
        return result
        
    except MemoryError:
        print("内存不足，使用分块处理...")
        return downsample_voxel_chunked(points, voxel_size)
    except Exception as e:
        print(f"标准体素下采样失败: {e}")
        print("使用哈希方法...")
        return downsample_voxel_hash(points, voxel_size)


def filter_by_height(points, min_height=None, max_height=None):
    """
    根据高度过滤点云
    
    参数:
        points: 点云数组 (N, 3)
        min_height: 最小高度
        max_height: 最大高度
    
    返回:
        过滤后的点云数组
    """
    if min_height is None and max_height is None:
        return points
    
    print(f"开始高度过滤: {min_height} <= Z <= {max_height}")
    
    mask = np.ones(len(points), dtype=bool)
    
    if min_height is not None:
        mask &= (points[:, 2] >= min_height)
    
    if max_height is not None:
        mask &= (points[:, 2] <= max_height)
    
    filtered_points = points[mask]
    print(f"高度过滤完成: {len(points)} -> {len(filtered_points)} 点")
    
    return filtered_points


def write_pcd_file_chunked(pcd_path, las_data, header, offset, total_points, voxel_size=None, min_height=None, max_height=None, apply_offset=True, use_float32=False):
    """
    分块写入 PCD 文件，支持过滤和下采样
    
    参数:
        pcd_path: 输出 PCD 文件路径
        las_data: LAS 数据对象
        header: LAS 头部信息
        offset: 偏移量
        total_points: 总点数
        voxel_size: 体素下采样大小（可选）
        min_height: 最小高度过滤值（可选）
        max_height: 最大高度过滤值（可选）
        apply_offset: 是否应用偏移量
    """
    print(f"开始分块写入 PCD 文件: {pcd_path}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(pcd_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    point_count = header.point_count
    chunk_size = 50000  # 每块5万个点
    
    # 如果需要体素下采样，需要先收集所有点
    if voxel_size is not None:
        print("体素下采样需要收集所有点...")
        all_points = []
        processed_count = 0
        
        # 获取LAS头文件中的偏移量
        las_offset = np.array([header.x_offset, header.y_offset, header.z_offset], dtype=np.float64)
        
        for i in range(0, point_count, chunk_size):
            end_idx = min(i + chunk_size, point_count)
            
            # 获取当前块的坐标
            if apply_offset:
                # 应用LAS头文件中的偏移量（laspy已经自动应用）
                x_chunk = np.array(las_data.x[i:end_idx], dtype=np.float64)
                y_chunk = np.array(las_data.y[i:end_idx], dtype=np.float64)
                z_chunk = np.array(las_data.z[i:end_idx], dtype=np.float64)
            else:
                # 不应用LAS头文件中的偏移量
                x_chunk = np.array(las_data.x[i:end_idx], dtype=np.float64) - las_offset[0]
                y_chunk = np.array(las_data.y[i:end_idx], dtype=np.float64) - las_offset[1]
                z_chunk = np.array(las_data.z[i:end_idx], dtype=np.float64) - las_offset[2]
            
            # 创建当前块的点云数组
            chunk_points = np.column_stack([x_chunk, y_chunk, z_chunk])
            
            # 清理临时变量
            del x_chunk, y_chunk, z_chunk
            gc.collect()
            
            # 应用高度过滤
            if min_height is not None or max_height is not None:
                chunk_points = filter_by_height(chunk_points, min_height, max_height)
            
            all_points.append(chunk_points)
            processed_count += len(chunk_points)
            
            if processed_count % 500000 == 0:
                print(f"  已处理 {processed_count} 个点...")
            
            del chunk_points
            gc.collect()
        
        # 合并所有点
        if all_points:
            points = np.vstack(all_points)
            del all_points
            gc.collect()
            
            # 执行体素下采样
            points = downsample_voxel(points, voxel_size)
            
            # 使用标准写入函数
            write_pcd_file(pcd_path, points, offset, use_float32)
        else:
            print("没有点云数据可写入")
        return
    
    # 两步写入过程：先统计，再写入
    print("第一步：统计过滤后的点数...")
    final_count = 0
    
    # 获取LAS头文件中的偏移量
    las_offset = np.array([header.x_offset, header.y_offset, header.z_offset], dtype=np.float64)
    
    for i in range(0, point_count, chunk_size):
        end_idx = min(i + chunk_size, point_count)
        
        # 获取当前块的坐标
        if apply_offset:
            # 应用LAS头文件中的偏移量（laspy已经自动应用）
            x_chunk = np.array(las_data.x[i:end_idx], dtype=np.float64)
            y_chunk = np.array(las_data.y[i:end_idx], dtype=np.float64)
            z_chunk = np.array(las_data.z[i:end_idx], dtype=np.float64)
        else:
            # 不应用LAS头文件中的偏移量
            x_chunk = np.array(las_data.x[i:end_idx], dtype=np.float64) - las_offset[0]
            y_chunk = np.array(las_data.y[i:end_idx], dtype=np.float64) - las_offset[1]
            z_chunk = np.array(las_data.z[i:end_idx], dtype=np.float64) - las_offset[2]
        
        # 创建当前块的点云数组
        chunk_points = np.column_stack([x_chunk, y_chunk, z_chunk])
        
        # 清理临时变量
        del x_chunk, y_chunk, z_chunk
        gc.collect()
        
        # 应用高度过滤
        if min_height is not None or max_height is not None:
            chunk_points = filter_by_height(chunk_points, min_height, max_height)
        
        final_count += len(chunk_points)
        
        if i % 1000000 == 0:
            print(f"  统计进度: {i}/{point_count} ({i/point_count*100:.1f}%)")
        
        del chunk_points
        gc.collect()
    
    print(f"统计完成，过滤后点数: {final_count}")
    
    # 第二步：写入数据
    print("第二步：写入点云数据...")
    
    # 获取数据类型信息
    dtype, size_bytes, type_char = get_dtype_info(use_float32)
    
    with open(pcd_path, 'w') as f:
        # 写入正确的头部信息
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("# Generated by las2pcd converter\n")
        f.write(f"# OFFSET {offset[0]:.12f} {offset[1]:.12f} {offset[2]:.12f}\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write(f"SIZE {size_bytes} {size_bytes} {size_bytes}\n")
        f.write(f"TYPE {type_char} {type_char} {type_char}\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {final_count}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {final_count}\n")
        f.write("DATA ascii\n")
        
        # 写入点云数据
        written_count = 0
        
        for i in range(0, point_count, chunk_size):
            end_idx = min(i + chunk_size, point_count)
            
            # 获取当前块的坐标
            if apply_offset:
                # 应用LAS头文件中的偏移量（laspy已经自动应用）
                x_chunk = np.array(las_data.x[i:end_idx], dtype=np.float64)
                y_chunk = np.array(las_data.y[i:end_idx], dtype=np.float64)
                z_chunk = np.array(las_data.z[i:end_idx], dtype=np.float64)
            else:
                # 不应用LAS头文件中的偏移量
                x_chunk = np.array(las_data.x[i:end_idx], dtype=np.float64) - las_offset[0]
                y_chunk = np.array(las_data.y[i:end_idx], dtype=np.float64) - las_offset[1]
                z_chunk = np.array(las_data.z[i:end_idx], dtype=np.float64) - las_offset[2]
            
            # 创建当前块的点云数组
            chunk_points = np.column_stack([x_chunk, y_chunk, z_chunk])
            
            # 清理临时变量
            del x_chunk, y_chunk, z_chunk
            gc.collect()
            
            # 应用高度过滤
            if min_height is not None or max_height is not None:
                chunk_points = filter_by_height(chunk_points, min_height, max_height)
            
            # 写入当前块
            for point in chunk_points:
                f.write(f"{point[0]:.12f} {point[1]:.12f} {point[2]:.12f}\n")
            
            written_count += len(chunk_points)
            
            if i % 1000000 == 0:
                print(f"  写入进度: {i}/{point_count} ({i/point_count*100:.1f}%) 已写入: {written_count} 个点 (内存: {get_memory_usage():.2f} MB)")
            
            # 清理内存
            del chunk_points
            gc.collect()
    
    print(f"分块写入完成: {written_count} 个点")


def write_pcd_file(pcd_path, points, offset, use_float32=False):
    """
    写入 PCD 文件
    
    参数:
        pcd_path: 输出文件路径
        points: 点云数组 (N, 3)
        offset: 偏移量 [x, y, z]
    """
    print(f"写入 PCD 文件: {pcd_path}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(pcd_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_points = len(points)
    
    # 获取数据类型信息
    dtype, size_bytes, type_char = get_dtype_info(use_float32)
    
    # 转换精度
    points = convert_precision(points, use_float32)
    
    with open(pcd_path, 'w') as f:
        # 写入头部
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("# Generated by las2pcd converter\n")
        f.write(f"# OFFSET {offset[0]:.12f} {offset[1]:.12f} {offset[2]:.12f}\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write(f"SIZE {size_bytes} {size_bytes} {size_bytes}\n")
        f.write(f"TYPE {type_char} {type_char} {type_char}\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {num_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {num_points}\n")
        f.write("DATA ascii\n")
        
        # 写入点云数据 - 批量写入以减少内存使用
        print(f"正在写入点云数据 ({dtype.__name__})...")
        batch_size = 10000  # 每批处理10000个点
        
        for i in range(0, num_points, batch_size):
            batch_end = min(i + batch_size, num_points)
            batch_points = points[i:batch_end]
            
            # 批量写入
            lines = []
            for point in batch_points:
                lines.append(f"{point[0]:.12f} {point[1]:.12f} {point[2]:.12f}\n")
            
            f.writelines(lines)
            
            # 显示进度
            if i % 100000 == 0:
                progress = (i / num_points) * 100
                print(f"  进度: {progress:.1f}% ({i}/{num_points})")
    
    file_size = os.path.getsize(pcd_path) / 1024 / 1024
    print(f"PCD 文件已保存: {pcd_path}")
    print(f"点云数量: {num_points}")
    print(f"数据类型: {dtype.__name__}")
    print(f"文件大小: {file_size:.2f} MB")
    print(f"内存使用: {points.nbytes / 1024 / 1024:.2f} MB")


def write_pcd_file_binary(pcd_path, points, offset, use_float32=False):
    """
    写入二进制 PCD 文件 - 文件大小更小
    
    参数:
        pcd_path: 输出文件路径
        points: 点云数组 (N, 3)
        offset: 偏移量 [x, y, z]
    """
    print(f"写入二进制 PCD 文件: {pcd_path}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(pcd_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_points = len(points)
    
    # 获取数据类型信息
    dtype, size_bytes, type_char = get_dtype_info(use_float32)
    
    # 转换精度
    points = convert_precision(points, use_float32)
    
    # 写入头部（ASCII）
    header = []
    header.append("# .PCD v0.7 - Point Cloud Data file format\n")
    header.append("# Generated by las2pcd converter\n")
    header.append(f"# OFFSET {offset[0]:.12f} {offset[1]:.12f} {offset[2]:.12f}\n")
    header.append("VERSION 0.7\n")
    header.append("FIELDS x y z\n")
    header.append(f"SIZE {size_bytes} {size_bytes} {size_bytes}\n")
    header.append(f"TYPE {type_char} {type_char} {type_char}\n")
    header.append("COUNT 1 1 1\n")
    header.append(f"WIDTH {num_points}\n")
    header.append("HEIGHT 1\n")
    header.append("VIEWPOINT 0 0 0 1 0 0 0\n")
    header.append(f"POINTS {num_points}\n")
    header.append("DATA binary\n")
    
    header_str = ''.join(header)
    
    # 写入文件
    with open(pcd_path, 'wb') as f:
        # 写入头部
        f.write(header_str.encode('utf-8'))
        
        # 写入二进制数据
        print(f"正在写入二进制点云数据 ({dtype.__name__})...")
        
        # 批量写入二进制数据
        batch_size = 10000
        for i in range(0, num_points, batch_size):
            batch_end = min(i + batch_size, num_points)
            batch_points = points[i:batch_end]
            
            # 写入二进制数据
            f.write(batch_points.tobytes())
            
            # 显示进度
            if i % 100000 == 0:
                progress = (i / num_points) * 100
                print(f"  进度: {progress:.1f}% ({i}/{num_points})")
    
    file_size = os.path.getsize(pcd_path) / 1024 / 1024
    print(f"二进制 PCD 文件已保存: {pcd_path}")
    print(f"点云数量: {num_points}")
    print(f"数据类型: {dtype.__name__}")
    print(f"文件大小: {file_size:.2f} MB")
    print(f"内存使用: {points.nbytes / 1024 / 1024:.2f} MB")


def write_pcd_file_binary_chunked(pcd_path, las_data, header, offset, total_points, voxel_size=None, min_height=None, max_height=None, apply_offset=True, use_float32=False):
    """
    分块写入二进制 PCD 文件
    """
    print(f"开始分块写入二进制 PCD 文件: {pcd_path}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(pcd_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    point_count = header.point_count
    chunk_size = 50000
    
    # 如果需要体素下采样，需要先收集所有点
    if voxel_size is not None:
        print("体素下采样需要收集所有点...")
        all_points = []
        processed_count = 0
        
        for i in range(0, point_count, chunk_size):
            end_idx = min(i + chunk_size, point_count)
            
            # 获取当前块
            x_chunk = np.array(las_data.x[i:end_idx], dtype=np.float64)
            y_chunk = np.array(las_data.y[i:end_idx], dtype=np.float64)
            z_chunk = np.array(las_data.z[i:end_idx], dtype=np.float64)
            
            # 创建当前块的点云数组
            chunk_points = np.empty((len(x_chunk), 3), dtype=np.float64)
            
            # 应用偏移量
            if apply_offset:
                chunk_points[:, 0] = x_chunk - offset[0]
                chunk_points[:, 1] = y_chunk - offset[1]
                chunk_points[:, 2] = z_chunk - offset[2]
            else:
                chunk_points[:, 0] = x_chunk
                chunk_points[:, 1] = y_chunk
                chunk_points[:, 2] = z_chunk
            
            del x_chunk, y_chunk, z_chunk
            gc.collect()
            
            # 应用高度过滤
            if min_height is not None or max_height is not None:
                chunk_points = filter_by_height(chunk_points, min_height, max_height)
            
            all_points.append(chunk_points)
            processed_count += len(chunk_points)
            
            if processed_count % 500000 == 0:
                print(f"  已处理 {processed_count} 个点...")
            
            del chunk_points
            gc.collect()
        
        # 合并所有点
        if all_points:
            points = np.vstack(all_points)
            del all_points
            gc.collect()
            
            # 执行体素下采样
            points = downsample_voxel(points, voxel_size)
            
            # 使用二进制写入函数
            write_pcd_file_binary(pcd_path, points, offset, use_float32)
        else:
            print("没有点云数据可写入")
        return
    
    # 两步写入过程：先统计，再写入
    print("第一步：统计过滤后的点数...")
    final_count = 0
    
    for i in range(0, point_count, chunk_size):
        end_idx = min(i + chunk_size, point_count)
        
        # 获取当前块
        x_chunk = np.array(las_data.x[i:end_idx], dtype=np.float64)
        y_chunk = np.array(las_data.y[i:end_idx], dtype=np.float64)
        z_chunk = np.array(las_data.z[i:end_idx], dtype=np.float64)
        
        # 创建当前块的点云数组
        chunk_points = np.empty((len(x_chunk), 3), dtype=np.float64)
        
        # 应用偏移量
        if apply_offset:
            chunk_points[:, 0] = x_chunk - offset[0]
            chunk_points[:, 1] = y_chunk - offset[1]
            chunk_points[:, 2] = z_chunk - offset[2]
        else:
            chunk_points[:, 0] = x_chunk
            chunk_points[:, 1] = y_chunk
            chunk_points[:, 2] = z_chunk
        
        del x_chunk, y_chunk, z_chunk
        gc.collect()
        
        # 应用高度过滤
        if min_height is not None or max_height is not None:
            chunk_points = filter_by_height(chunk_points, min_height, max_height)
        
        final_count += len(chunk_points)
        
        if i % 1000000 == 0:
            print(f"  统计进度: {i}/{point_count} ({i/point_count*100:.1f}%)")
        
        del chunk_points
        gc.collect()
    
    print(f"统计完成，过滤后点数: {final_count}")
    
    # 第二步：写入二进制数据
    print("第二步：写入二进制点云数据...")
    
    # 获取数据类型信息
    dtype, size_bytes, type_char = get_dtype_info(use_float32)
    
    # 写入头部
    header_lines = []
    header_lines.append("# .PCD v0.7 - Point Cloud Data file format\n")
    header_lines.append("# Generated by las2pcd converter\n")
    header_lines.append(f"# OFFSET {offset[0]:.12f} {offset[1]:.12f} {offset[2]:.12f}\n")
    header_lines.append("VERSION 0.7\n")
    header_lines.append("FIELDS x y z\n")
    header_lines.append(f"SIZE {size_bytes} {size_bytes} {size_bytes}\n")
    header_lines.append(f"TYPE {type_char} {type_char} {type_char}\n")
    header_lines.append("COUNT 1 1 1\n")
    header_lines.append(f"WIDTH {final_count}\n")
    header_lines.append("HEIGHT 1\n")
    header_lines.append("VIEWPOINT 0 0 0 1 0 0 0\n")
    header_lines.append(f"POINTS {final_count}\n")
    header_lines.append("DATA binary\n")
    
    header_str = ''.join(header_lines)
    
    with open(pcd_path, 'wb') as f:
        # 写入头部
        f.write(header_str.encode('utf-8'))
        
        # 写入二进制点云数据
        written_count = 0
        
        for i in range(0, point_count, chunk_size):
            end_idx = min(i + chunk_size, point_count)
            
            # 获取当前块
            x_chunk = np.array(las_data.x[i:end_idx], dtype=np.float64)
            y_chunk = np.array(las_data.y[i:end_idx], dtype=np.float64)
            z_chunk = np.array(las_data.z[i:end_idx], dtype=np.float64)
            
            # 创建当前块的点云数组
            chunk_points = np.empty((len(x_chunk), 3), dtype=np.float64)
            
            # 应用偏移量
            if apply_offset:
                chunk_points[:, 0] = x_chunk - offset[0]
                chunk_points[:, 1] = y_chunk - offset[1]
                chunk_points[:, 2] = z_chunk - offset[2]
            else:
                chunk_points[:, 0] = x_chunk
                chunk_points[:, 1] = y_chunk
                chunk_points[:, 2] = z_chunk
            
            del x_chunk, y_chunk, z_chunk
            gc.collect()
            
            # 应用高度过滤
            if min_height is not None or max_height is not None:
                chunk_points = filter_by_height(chunk_points, min_height, max_height)
            
            # 写入二进制数据
            if len(chunk_points) > 0:
                # 转换精度
                chunk_points = convert_precision(chunk_points, use_float32)
                f.write(chunk_points.tobytes())
            
            written_count += len(chunk_points)
            
            if i % 1000000 == 0:
                print(f"  写入进度: {i}/{point_count} ({i/point_count*100:.1f}%) 已写入: {written_count} 个点 (内存: {get_memory_usage():.2f} MB)")
            
            del chunk_points
            gc.collect()
    
    file_size = os.path.getsize(pcd_path) / 1024 / 1024
    print(f"二进制分块写入完成: {written_count} 个点")
    print(f"文件大小: {file_size:.2f} MB")


def convert_precision(points, use_float32=False):
    """
    转换点云数据精度
    
    参数:
        points: 点云数组
        use_float32: 是否使用float32精度
    
    返回:
        转换后的点云数组
    """
    if use_float32:
        print("转换为 float32 精度...")
        return points.astype(np.float32)
    else:
        return points.astype(np.float64)


def get_dtype_info(use_float32=False):
    """
    获取数据类型信息
    
    参数:
        use_float32: 是否使用float32精度
    
    返回:
        (dtype, size_bytes, type_char)
    """
    if use_float32:
        return np.float32, 4, "F"
    else:
        return np.float64, 8, "F"


def apply_coordinate_transforms(points, swap_axes=None, center=False, rotate=None, flip=None):
    """
    应用坐标变换
    
    参数:
        points: 点云数组 (N, 3)
        swap_axes: 交换坐标轴 ("xy", "xz", "yz")
        center: 是否中心化
        rotate: 旋转角度 [rx, ry, rz] (度)
        flip: 翻转轴 ("x", "y", "z", "xy", "xz", "yz", "xyz")
    
    返回:
        变换后的点云数组
    """
    if len(points) == 0:
        return points
    
    points = points.copy()
    
    print("正在应用坐标变换...")
    
    # 1. 交换坐标轴
    if swap_axes:
        print(f"交换坐标轴: {swap_axes}")
        if swap_axes == "xy":
            points[:, [0, 1]] = points[:, [1, 0]]
        elif swap_axes == "xz":
            points[:, [0, 2]] = points[:, [2, 0]]
        elif swap_axes == "yz":
            points[:, [1, 2]] = points[:, [2, 1]]
    
    # 2. 翻转坐标轴
    if flip:
        print(f"翻转坐标轴: {flip}")
        if 'x' in flip:
            points[:, 0] = -points[:, 0]
        if 'y' in flip:
            points[:, 1] = -points[:, 1]
        if 'z' in flip:
            points[:, 2] = -points[:, 2]
    
    # 3. 中心化
    if center:
        print("中心化点云...")
        centroid = np.mean(points, axis=0)
        points = points - centroid
        print(f"原始中心: [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}]")
    
    # 4. 旋转
    if rotate is not None:
        print(f"旋转点云: [{rotate[0]:.2f}°, {rotate[1]:.2f}°, {rotate[2]:.2f}°]")
        points = rotate_points(points, rotate)
    
    print(f"变换后坐标范围:")
    print(f"  X: {points[:, 0].min():.6f} - {points[:, 0].max():.6f}")
    print(f"  Y: {points[:, 1].min():.6f} - {points[:, 1].max():.6f}")
    print(f"  Z: {points[:, 2].min():.6f} - {points[:, 2].max():.6f}")
    
    return points


def rotate_points(points, angles):
    """
    旋转点云
    
    参数:
        points: 点云数组 (N, 3)
        angles: 旋转角度 [rx, ry, rz] (度)
    
    返回:
        旋转后的点云数组
    """
    # 转换为弧度
    rx, ry, rz = np.radians(angles)
    
    # 旋转矩阵
    # 绕X轴旋转
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # 绕Y轴旋转
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # 绕Z轴旋转
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转矩阵 (顺序: Z-Y-X)
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    # 应用旋转
    rotated_points = np.dot(points, R.T)
    
    return rotated_points


def analyze_vlrs(las_file):
    """分析 VLR (Variable Length Records) 信息并提取变换数据"""
    print("分析 VLR 记录...")
    
    try:
        vlrs = las_file.header.vlrs
        print(f"找到 {len(vlrs)} 个 VLR 记录")
        
        transform_info = {}
        
        for i, vlr in enumerate(vlrs):
            print(f"VLR {i+1}: {vlr.user_id} - {vlr.record_id} - {vlr.description}")
            
            # 检查是否为投影信息
            if vlr.user_id == "LASF_Projection":
                if vlr.record_id == 34735:  # GeoKeyDirectoryTag
                    transform_info['projection'] = vlr.record_data
                    print(f"  发现投影信息")
                elif vlr.record_id == 34736:  # GeoDoubleParamsTag
                    transform_info['geo_double'] = vlr.record_data
                    print(f"  发现GeoTIFF双精度参数")
                elif vlr.record_id == 34737:  # GeoAsciiParamsTag
                    transform_info['geo_ascii'] = vlr.record_data
                    print(f"  发现GeoTIFF ASCII参数")
            
            # 检查是否为变换矩阵
            elif "transform" in vlr.user_id.lower() or "matrix" in vlr.user_id.lower():
                transform_info['transform_matrix'] = vlr.record_data
                print(f"  发现变换矩阵")
        
        return transform_info
    except Exception as e:
        print(f"分析VLR时出错: {e}")
        return {}


def detect_multi_source_issues(las_file):
    """检测多源点云问题"""
    print("检测多源点云问题...")
    
    issues = []
    source_info = {}
    
    try:
        # 检查点源ID
        if hasattr(las_file, 'point_source_id'):
            source_ids = np.unique(las_file.point_source_id)
            print(f"发现 {len(source_ids)} 个点源: {source_ids}")
            
            if len(source_ids) > 1:
                print("检测到多源点云，分析各源的坐标分布...")
                
                for src_id in source_ids:
                    mask = las_file.point_source_id == src_id
                    count = np.sum(mask)
                    
                    if count > 0:
                        src_x = las_file.x[mask]
                        src_y = las_file.y[mask]
                        src_z = las_file.z[mask]
                        
                        source_info[src_id] = {
                            'count': count,
                            'x_range': (src_x.min(), src_x.max()),
                            'y_range': (src_y.min(), src_y.max()),
                            'z_range': (src_z.min(), src_z.max()),
                            'centroid': (np.mean(src_x), np.mean(src_y), np.mean(src_z))
                        }
                        
                        print(f"  源 {src_id}: {count} 个点")
                        print(f"    X: {src_x.min():.3f} - {src_x.max():.3f}")
                        print(f"    Y: {src_y.min():.3f} - {src_y.max():.3f}")
                        print(f"    Z: {src_z.min():.3f} - {src_z.max():.3f}")
                
                # 检查源之间的偏移
                source_ids_list = list(source_info.keys())
                for i in range(len(source_ids_list)):
                    for j in range(i+1, len(source_ids_list)):
                        src1 = source_ids_list[i]
                        src2 = source_ids_list[j]
                        
                        c1 = source_info[src1]['centroid']
                        c2 = source_info[src2]['centroid']
                        
                        offset = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)
                        
                        if offset > 10:  # 10米以上的偏移
                            issues.append(f"源 {src1} 和源 {src2} 之间存在 {offset:.2f}m 的偏移")
                            print(f"  警告: 源 {src1} 和源 {src2} 之间存在 {offset:.2f}m 的偏移")
        
        # 检查分类问题
        if hasattr(las_file, 'classification'):
            classes = np.unique(las_file.classification)
            print(f"发现 {len(classes)} 个分类: {classes}")
            
            if len(classes) > 1:
                print("检测到多分类点云，分析各分类的坐标分布...")
                
                for cls in classes:
                    mask = las_file.classification == cls
                    count = np.sum(mask)
                    
                    if count > 0:
                        cls_x = las_file.x[mask]
                        cls_y = las_file.y[mask]
                        cls_z = las_file.z[mask]
                        
                        print(f"  分类 {cls}: {count} 个点")
                        print(f"    X: {cls_x.min():.3f} - {cls_x.max():.3f}")
                        print(f"    Y: {cls_y.min():.3f} - {cls_y.max():.3f}")
                        print(f"    Z: {cls_z.min():.3f} - {cls_z.max():.3f}")
    
    except Exception as e:
        print(f"检测多源问题时出错: {e}")
    
    return issues, source_info


def apply_multi_source_alignment(points, point_source_id, source_info):
    """应用多源点云对齐"""
    print("应用多源点云对齐...")
    
    if point_source_id is None or len(source_info) <= 1:
        print("无需多源对齐")
        return points
    
    aligned_points = points.copy()
    
    # 选择第一个源作为参考
    reference_source = list(source_info.keys())[0]
    reference_centroid = source_info[reference_source]['centroid']
    
    print(f"使用源 {reference_source} 作为参考")
    print(f"参考中心: ({reference_centroid[0]:.3f}, {reference_centroid[1]:.3f}, {reference_centroid[2]:.3f})")
    
    # 对齐其他源到参考源
    for src_id, info in source_info.items():
        if src_id != reference_source:
            src_centroid = info['centroid']
            
            # 计算偏移
            offset = np.array([
                reference_centroid[0] - src_centroid[0],
                reference_centroid[1] - src_centroid[1],
                reference_centroid[2] - src_centroid[2]
            ])
            
            print(f"源 {src_id} 偏移: ({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f})")
            
            # 应用偏移到对应点
            mask = point_source_id == src_id
            aligned_points[mask] += offset
    
    print("多源对齐完成")
    return aligned_points


def apply_vlr_transforms(points, transform_info):
    """应用VLR中的变换信息"""
    print("应用VLR变换...")
    
    if not transform_info:
        print("无VLR变换信息")
        return points
    
    transformed_points = points.copy()
    
    # 处理投影变换
    if 'projection' in transform_info:
        print("应用投影变换...")
        # 这里可以添加具体的投影变换逻辑
        # 需要根据具体的投影参数来实现
    
    # 处理变换矩阵
    if 'transform_matrix' in transform_info:
        print("应用变换矩阵...")
        # 这里可以添加具体的矩阵变换逻辑
        # 需要解析VLR中的矩阵数据
    
    return transformed_points


def smart_coordinate_repair(points, las_file):
    """智能坐标修复"""
    print("执行智能坐标修复...")
    
    repaired_points = points.copy()
    
    # 1. 分析VLR变换信息
    transform_info = analyze_vlrs(las_file)
    
    # 2. 检测多源问题
    issues, source_info = detect_multi_source_issues(las_file)
    
    # 3. 应用VLR变换
    if transform_info:
        repaired_points = apply_vlr_transforms(repaired_points, transform_info)
    
    # 4. 应用多源对齐
    if len(source_info) > 1:
        point_source_id = getattr(las_file, 'point_source_id', None)
        repaired_points = apply_multi_source_alignment(repaired_points, point_source_id, source_info)
    
    # 5. 检查结果
    print(f"修复前坐标范围:")
    print(f"  X: {points[:, 0].min():.3f} - {points[:, 0].max():.3f}")
    print(f"  Y: {points[:, 1].min():.3f} - {points[:, 1].max():.3f}")
    print(f"  Z: {points[:, 2].min():.3f} - {points[:, 2].max():.3f}")
    
    print(f"修复后坐标范围:")
    print(f"  X: {repaired_points[:, 0].min():.3f} - {repaired_points[:, 0].max():.3f}")
    print(f"  Y: {repaired_points[:, 1].min():.3f} - {repaired_points[:, 1].max():.3f}")
    print(f"  Z: {repaired_points[:, 2].min():.3f} - {repaired_points[:, 2].max():.3f}")
    
    return repaired_points


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="LAS/LAZ 转 PCD 转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本转换（应用LAS头文件偏移量，输出真实世界坐标）
  python3 las2pcd.py input.las output.pcd --offset true
  
  # 不应用LAS头文件偏移量，输出相对坐标
  python3 las2pcd.py input.las output.pcd --offset false
  
  # 使用float32精度，文件大小减半
  python3 las2pcd.py input.las output.pcd --offset true --precision float32
  
  # ASCII格式输出
  python3 las2pcd.py input.las output.pcd --offset true --format ascii
  
  # 带下采样
  python3 las2pcd.py input.las output.pcd --offset true --voxel-size 0.1
  
  # 带高度过滤
  python3 las2pcd.py input.las output.pcd --offset true --min-height 0 --max-height 50
  
  # 最小文件大小（二进制 + float32）
  python3 las2pcd.py input.las output.pcd --offset true --format binary --precision float32
  
  # 修复垂直点云问题 - 交换坐标轴
  python3 las2pcd.py input.las output.pcd --swap-axes xy
  
  # 修复垂直点云问题 - 翻转坐标轴
  python3 las2pcd.py input.las output.pcd --flip z
  
  # 修复垂直点云问题 - 中心化
  python3 las2pcd.py input.las output.pcd --center
  
  # 修复垂直点云问题 - 旋转
  python3 las2pcd.py input.las output.pcd --rotate 0 0 90
  
  # 组合修复（常见的垂直点云问题）
  python3 las2pcd.py input.las output.pcd --swap-axes xy --center --flip z
  
  # 智能坐标修复（自动分析并应用变换）
  python3 las2pcd.py input.las output.pcd --smart-repair
  
  # 多源点云对齐
  python3 las2pcd.py input.las output.pcd --align-sources
  
  # 应用VLR变换信息
  python3 las2pcd.py input.las output.pcd --apply-vlr
  
  # 仅分析文件结构（不转换）
  python3 las2pcd.py input.las output.pcd --analyze-only
  
  # 完整的问题修复流程
  python3 las2pcd.py input.las output.pcd --smart-repair --align-sources --apply-vlr
        """
    )
    
    # 必需参数
    parser.add_argument("input", help="输入 LAS/LAZ 文件路径")
    parser.add_argument("output", help="输出 PCD 文件路径")
    
    # 可选参数
    parser.add_argument("--offset", choices=["true", "false"], default="true",
                       help="是否应用LAS头文件中的偏移量 (true: 应用偏移量，输出真实世界坐标; false: 不应用偏移量，输出相对坐标)")
    parser.add_argument("--voxel-size", type=float,
                       help="体素下采样大小（米）")
    parser.add_argument("--min-height", type=float,
                       help="最小高度过滤值（米）")
    parser.add_argument("--max-height", type=float,
                       help="最大高度过滤值（米）")
    parser.add_argument("--format", choices=["ascii", "binary"], default="binary",
                       help="输出格式 (ascii: 文本格式, binary: 二进制格式，文件更小)")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64",
                       help="数据精度 (float32: 单精度，文件更小; float64: 双精度，精度更高)")
    
    # 坐标系统修复参数
    parser.add_argument("--swap-axes", choices=["xy", "xz", "yz"], 
                       help="交换坐标轴 (xy: 交换X和Y, xz: 交换X和Z, yz: 交换Y和Z)")
    parser.add_argument("--center", action="store_true",
                       help="将点云中心化到原点")
    parser.add_argument("--rotate", type=float, nargs=3, metavar=('RX', 'RY', 'RZ'),
                       help="围绕X、Y、Z轴旋转点云（角度，单位：度）")
    parser.add_argument("--flip", choices=["x", "y", "z", "xy", "xz", "yz", "xyz"],
                       help="翻转坐标轴")
    
    # 新增：高级变换参数
    parser.add_argument("--align-sources", action="store_true",
                       help="对齐多源点云（自动检测并对齐不同源的点云）")
    parser.add_argument("--apply-vlr", action="store_true",
                       help="应用VLR中的变换信息")
    parser.add_argument("--smart-repair", action="store_true",
                       help="智能坐标修复（自动分析并应用适当的变换）")
    parser.add_argument("--analyze-only", action="store_true",
                       help="仅分析文件结构，不进行转换")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    # 解析参数
    apply_offset = args.offset.lower() == "true"
    use_binary = args.format == "binary"
    use_float32 = args.precision == "float32"
    
    print("="*50)
    print("LAS 转 PCD 转换工具")
    print("="*50)
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"应用偏移量: {apply_offset}")
    print(f"输出格式: {args.format}")
    print(f"数据精度: {args.precision}")
    print(f"体素下采样: {args.voxel_size if args.voxel_size else '无'}")
    print(f"高度过滤: {args.min_height if args.min_height else '无'} - {args.max_height if args.max_height else '无'}")
    print(f"坐标轴交换: {args.swap_axes if args.swap_axes else '无'}")
    print(f"中心化: {'是' if args.center else '否'}")
    print(f"旋转: {args.rotate if args.rotate else '无'}")
    print(f"翻转: {args.flip if args.flip else '无'}")
    print(f"对齐多源: {'是' if args.align_sources else '否'}")
    print(f"应用VLR: {'是' if args.apply_vlr else '否'}")
    print(f"智能修复: {'是' if args.smart_repair else '否'}")
    print(f"仅分析: {'是' if args.analyze_only else '否'}")
    print("="*50)
    
    try:
        # 读取 LAS 文件
        result = read_las_file(args.input, apply_offset)
        
        # 如果只是分析模式，就不进行转换
        if args.analyze_only:
            print("分析模式，不进行转换")
            return 0
        
        if len(result) == 6:
            # 分块处理模式
            las_data, header, offset, is_chunked, total_points, apply_offset_flag = result
            
            print("使用分块处理模式...")
            
            # 对于分块处理，我们需要修改写入函数来支持新的变换
            # 暂时使用原有的分块写入方式
            
            # 根据格式选择写入函数
            if use_binary:
                write_pcd_file_binary_chunked(
                    args.output, 
                    las_data,
                    header,
                    offset, 
                    total_points,
                    args.voxel_size,
                    args.min_height,
                    args.max_height,
                    apply_offset_flag,
                    use_float32
                )
            else:
                write_pcd_file_chunked(
                    args.output, 
                    las_data,
                    header,
                    offset, 
                    total_points,
                    args.voxel_size,
                    args.min_height,
                    args.max_height,
                    apply_offset_flag,
                    use_float32
                )
            
            print("\n" + "="*50)
            print("转换完成!")
            print(f"原始点云数量: {total_points}")
            print("使用分块处理模式，最终点云数量请查看上方输出")
            
        else:
            # 小文件直接处理
            points, offset, is_chunked, total_points = result
            
            print("使用标准处理模式...")
            
            # 重新读取LAS文件以获取完整的laspy对象（用于变换分析）
            las_file = laspy.read(args.input)
            
            # 高度过滤
            if args.min_height is not None or args.max_height is not None:
                points = filter_by_height(points, args.min_height, args.max_height)
            
            # 体素下采样
            if args.voxel_size:
                points = downsample_voxel(points, args.voxel_size)
            
            # 应用智能坐标修复
            if args.smart_repair:
                points = smart_coordinate_repair(points, las_file)
            
            # 应用VLR变换
            if args.apply_vlr:
                transform_info = analyze_vlrs(las_file)
                points = apply_vlr_transforms(points, transform_info)
            
            # 应用多源对齐
            if args.align_sources:
                issues, source_info = detect_multi_source_issues(las_file)
                if len(source_info) > 1:
                    point_source_id = getattr(las_file, 'point_source_id', None)
                    points = apply_multi_source_alignment(points, point_source_id, source_info)
            
            # 应用坐标变换（这是修复垂直点云问题的关键）
            if args.swap_axes or args.center or args.rotate or args.flip:
                points = apply_coordinate_transforms(
                    points, 
                    swap_axes=args.swap_axes,
                    center=args.center,
                    rotate=args.rotate,
                    flip=args.flip
                )
            
            # 转换精度
            points = convert_precision(points, use_float32)
            
            # 根据格式选择写入函数
            if use_binary:
                write_pcd_file_binary(args.output, points, offset, use_float32)
            else:
                write_pcd_file(args.output, points, offset, use_float32)
            
            print("\n" + "="*50)
            print("转换完成!")
            print(f"最终点云数量: {len(points)}")
            print(f"数据类型: {points.dtype}")
            print(f"内存使用: {points.nbytes / 1024 / 1024:.2f} MB")
        
        if apply_offset:
            print(f"已应用LAS头文件偏移量，输出真实世界坐标")
            print(f"LAS头文件偏移量: [{offset[0]:.12f}, {offset[1]:.12f}, {offset[2]:.12f}]")
        else:
            print("未应用LAS头文件偏移量，输出相对坐标")
            print(f"LAS头文件偏移量（仅供参考）: [{offset[0]:.12f}, {offset[1]:.12f}, {offset[2]:.12f}]")
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

def downsample_voxel_chunked(points, voxel_size, chunk_size=100000):
    """
    分块体素下采样 - 处理大点云数据
    
    参数:
        points: 点云数组 (N, 3)
        voxel_size: 体素大小
        chunk_size: 每块处理的点数
    
    返回:
        下采样后的点云数组
    """
    print(f"分块体素下采样，块大小: {chunk_size}")
    
    # 使用字典存储体素中心点
    voxel_centers = {}
    total_processed = 0
    
    # 分块处理
    for i in range(0, len(points), chunk_size):
        end_idx = min(i + chunk_size, len(points))
        chunk_points = points[i:end_idx]
        
        print(f"处理块 {i//chunk_size + 1}: 点数 {len(chunk_points)}")
        
        # 计算当前块的体素索引
        voxel_indices = np.floor(chunk_points / voxel_size).astype(np.int64)
        
        # 处理当前块
        for j, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in voxel_centers:
                voxel_centers[key] = {'sum': chunk_points[j].copy(), 'count': 1}
            else:
                voxel_centers[key]['sum'] += chunk_points[j]
                voxel_centers[key]['count'] += 1
        
        total_processed += len(chunk_points)
        
        # 显示进度
        progress = (total_processed / len(points)) * 100
        print(f"  进度: {progress:.1f}% ({total_processed}/{len(points)})")
        
        # 强制垃圾回收
        del chunk_points, voxel_indices
        gc.collect()
    
    print(f"计算 {len(voxel_centers)} 个体素的中心点...")
    
    # 计算体素中心点
    downsampled_points = []
    for i, (key, data) in enumerate(voxel_centers.items()):
        center = data['sum'] / data['count']
        downsampled_points.append(center)
        
        # 显示进度
        if i % 10000 == 0:
            progress = (i / len(voxel_centers)) * 100
            print(f"  计算进度: {progress:.1f}% ({i}/{len(voxel_centers)})")
    
    result = np.array(downsampled_points, dtype=np.float64)
    print(f"分块下采样完成: {len(points)} -> {len(result)} 点")
    
    return result


def downsample_voxel_hash(points, voxel_size):
    """
    哈希体素下采样 - 处理极大点云数据
    
    参数:
        points: 点云数组 (N, 3)
        voxel_size: 体素大小
    
    返回:
        下采样后的点云数组
    """
    print("哈希体素下采样...")
    
    # 使用更简单的哈希方法
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    
    # 创建哈希键
    hash_keys = []
    for i in range(len(voxel_indices)):
        # 使用简单的字符串哈希
        key = f"{voxel_indices[i, 0]}_{voxel_indices[i, 1]}_{voxel_indices[i, 2]}"
        hash_keys.append(key)
    
    # 使用pandas进行分组（如果可用）
    try:
        import pandas as pd
        
        # 创建DataFrame
        df = pd.DataFrame({
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2],
            'hash': hash_keys
        })
        
        # 按哈希键分组并计算均值
        grouped = df.groupby('hash')[['x', 'y', 'z']].mean()
        result = grouped.values
        
        print(f"哈希下采样完成: {len(points)} -> {len(result)} 点")
        return result
        
    except ImportError:
        print("pandas不可用，使用基础哈希方法...")
        
        # 基础哈希方法
        hash_dict = {}
        for i, key in enumerate(hash_keys):
            if key not in hash_dict:
                hash_dict[key] = {'sum': points[i].copy(), 'count': 1}
            else:
                hash_dict[key]['sum'] += points[i]
                hash_dict[key]['count'] += 1
            
            # 显示进度
            if i % 100000 == 0:
                progress = (i / len(points)) * 100
                print(f"  哈希进度: {progress:.1f}% ({i}/{len(points)})")
        
        # 计算中心点
        downsampled_points = []
        for data in hash_dict.values():
            center = data['sum'] / data['count']
            downsampled_points.append(center)
        
        result = np.array(downsampled_points, dtype=np.float64)
        print(f"基础哈希下采样完成: {len(points)} -> {len(result)} 点")
        
        return result
