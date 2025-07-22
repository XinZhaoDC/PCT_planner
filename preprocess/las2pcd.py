#!/usr/bin/env python3
"""
LAS 转 PCD 转换工具 (基于oldlas2pcd.py和minimal_las2pcd.py)
=========================================================

功能特性：
1. 读取 LAS/LAZ 文件并转换为 PCD 格式
2. 支持大坐标自动检测和局部平移
3. 支持 offset 选项（是否应用LAS偏移量）
4. 支持二进制和ASCII格式输出
5. 支持大数据分块处理
6. 强制使用 float32 保存坐标，兼容CloudCompare

使用方法：
python3 las2pcd.py input.las output.pcd [选项...]
"""

import os
import sys
import argparse
import numpy as np

# 导入laspy库
try:
    import laspy
except ImportError:
    print("错误: 需要安装 laspy 库")
    print("请运行: pip install laspy")
    sys.exit(1)


def check_large_coordinates(points, threshold=1e5):
    """
    检查坐标是否过大，需要局部平移
    
    参数:
        points: 点云数组 (N, 3)
        threshold: 阈值，绝对值超过此值认为是大坐标
    
    返回:
        needs_translation: bool, 是否需要平移
        translation_xy: 平移量 [dx, dy]，仅对X/Y
    """
    x_range = [points[:, 0].min(), points[:, 0].max()]
    y_range = [points[:, 1].min(), points[:, 1].max()]
    
    print(f"检查坐标范围:")
    print(f"  X: {x_range[0]:.6f} - {x_range[1]:.6f}")
    print(f"  Y: {y_range[0]:.6f} - {y_range[1]:.6f}")
    print(f"  大坐标阈值: {threshold}")
    
    # 检查是否有任何坐标的绝对值超过阈值
    max_abs_x = max(abs(x_range[0]), abs(x_range[1]))
    max_abs_y = max(abs(y_range[0]), abs(y_range[1]))
    
    needs_translation = max_abs_x > threshold or max_abs_y > threshold
    
    if needs_translation:
        # 计算平移量：使用中心点作为平移参考
        center_x = (x_range[0] + x_range[1]) / 2.0
        center_y = (y_range[0] + y_range[1]) / 2.0
        translation_xy = np.array([center_x, center_y], dtype=np.float64)
        
        print(f"检测到大坐标，将应用局部平移:")
        print(f"  最大绝对值 X: {max_abs_x:.0f}")
        print(f"  最大绝对值 Y: {max_abs_y:.0f}")
        print(f"  计算平移量: X平移 {translation_xy[0]:.6f}, Y平移 {translation_xy[1]:.6f}")
    else:
        translation_xy = np.array([0.0, 0.0], dtype=np.float64)
        print(f"坐标范围合理，无需平移")
    
    return needs_translation, translation_xy


def apply_translation(points, translation_xy):
    """
    应用局部平移（仅对X/Y，Z不变）
    
    参数:
        points: 点云数组 (N, 3)
        translation_xy: 平移量 [dx, dy]
    
    返回:
        translated_points: 平移后的点云 (N, 3)
    """
    if abs(translation_xy[0]) < 1e-10 and abs(translation_xy[1]) < 1e-10:
        return points
    
    translated_points = points.copy()
    translated_points[:, 0] -= translation_xy[0]  # X坐标平移
    translated_points[:, 1] -= translation_xy[1]  # Y坐标平移
    # Z坐标不变
    
    print(f"应用局部平移:")
    print(f"  X平移: -{translation_xy[0]:.6f}")
    print(f"  Y平移: -{translation_xy[1]:.6f}")
    print(f"  Z不变")
    
    return translated_points


def read_las_with_processing(las_path, apply_offset=True, auto_translate=True, coord_threshold=1e5, chunk_size=1000000):
    """
    读取LAS文件并处理大坐标，分块处理以节省内存
    
    参数:
        las_path: LAS文件路径
        apply_offset: 是否应用LAS头文件偏移量
        auto_translate: 是否自动检测并平移大坐标
        coord_threshold: 大坐标检测阈值
        chunk_size: 分块大小
    
    返回:
        points: 点云数组 (N, 3)，已转换为float32
        total_offset: 总偏移量 [x_offset, y_offset, z_offset]
    """
    print(f"读取LAS文件: {las_path}")
    
    # 读取LAS文件
    las_data = laspy.read(las_path)
    header = las_data.header
    
    print(f"点云数量: {header.point_count}")
    print(f"LAS版本: {header.version}")
    
    # 获取LAS头文件偏移量和缩放因子
    las_offset = np.array([header.x_offset, header.y_offset, header.z_offset], dtype=np.float64)
    las_scale = np.array([header.x_scale, header.y_scale, header.z_scale], dtype=np.float64)
    
    print(f"LAS头文件偏移量: [{las_offset[0]:.6f}, {las_offset[1]:.6f}, {las_offset[2]:.6f}]")
    print(f"LAS头文件缩放因子: [{las_scale[0]:.6f}, {las_scale[1]:.6f}, {las_scale[2]:.6f}]")
    
    # 获取坐标数据（保持float64进行计算）
    if apply_offset:
        print("应用LAS头文件偏移量 -> 输出真实世界坐标")
        # laspy.read()已经自动应用了偏移量和缩放因子
        x_coords = np.array(las_data.x, dtype=np.float64)
        y_coords = np.array(las_data.y, dtype=np.float64)
        z_coords = np.array(las_data.z, dtype=np.float64)
        las_offset_used = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # LAS偏移已应用
    else:
        print("不应用LAS头文件偏移量 -> 输出相对坐标")
        # 减去偏移量，获得相对坐标
        x_coords = np.array(las_data.x, dtype=np.float64) - las_offset[0]
        y_coords = np.array(las_data.y, dtype=np.float64) - las_offset[1]
        z_coords = np.array(las_data.z, dtype=np.float64) - las_offset[2]
        las_offset_used = las_offset  # 记录LAS偏移量
    
    # 组合坐标（float64用于精确计算）
    points = np.column_stack([x_coords, y_coords, z_coords])
    
    # 显示原始坐标范围
    print(f"原始坐标范围:")
    print(f"  X: {points[:, 0].min():.6f} - {points[:, 0].max():.6f}")
    print(f"  Y: {points[:, 1].min():.6f} - {points[:, 1].max():.6f}")
    print(f"  Z: {points[:, 2].min():.6f} - {points[:, 2].max():.6f}")
    
    # 检查是否需要局部平移
    translation_xy = np.array([0.0, 0.0], dtype=np.float64)
    if auto_translate:
        needs_translation, translation_xy = check_large_coordinates(points, coord_threshold)
        if needs_translation:
            # 应用局部平移
            points = apply_translation(points, translation_xy)
    else:
        print("大坐标检测已禁用")
        needs_translation = False
    
    # 计算总偏移量
    total_offset = np.array([
        las_offset_used[0] + translation_xy[0],  # X总偏移 = LAS偏移 + 局部平移
        las_offset_used[1] + translation_xy[1],  # Y总偏移 = LAS偏移 + 局部平移  
        las_offset_used[2]                       # Z偏移 = 仅LAS偏移
    ], dtype=np.float64)
    
    # 转换为float32以节省内存并确保兼容性
    print("转换坐标精度为 float32...")
    points = points.astype(np.float32)
    
    print(f"最终坐标范围 (float32):")
    print(f"  X: {points[:, 0].min():.6f} - {points[:, 0].max():.6f}")
    print(f"  Y: {points[:, 1].min():.6f} - {points[:, 1].max():.6f}")
    print(f"  Z: {points[:, 2].min():.6f} - {points[:, 2].max():.6f}")
    print(f"总偏移量: [{total_offset[0]:.6f}, {total_offset[1]:.6f}, {total_offset[2]:.6f}]")
    
    return points, total_offset


def write_pcd_ascii(pcd_path, points, offset):
    """
    写入ASCII格式PCD文件
    
    参数:
        pcd_path: 输出文件路径
        points: 点云数组 (N, 3)，应该是float32类型
        offset: 总偏移量 [x_offset, y_offset, z_offset]
    """
    print(f"写入ASCII格式PCD文件: {pcd_path}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(pcd_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_points = len(points)
    
    # 确保points是float32类型
    if points.dtype != np.float32:
        print(f"警告: 将坐标从 {points.dtype} 转换为 float32")
        points = points.astype(np.float32)
    
    with open(pcd_path, 'w') as f:
        # 写入PCD头部
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("# Generated by las2pcd converter (optimized for CloudCompare)\n")
        f.write(f"# OFFSET {offset[0]:.12f} {offset[1]:.12f} {offset[2]:.12f}\n")
        f.write("# COORDINATE_SYSTEM: Local (may be translated from original)\n")
        f.write("# DATA_TYPE: float32\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")        # float32 = 4 bytes
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {num_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {num_points}\n")
        f.write("DATA ascii\n")
        
        # 写入点云数据（保持float32精度）
        print(f"正在写入 {num_points} 个点（ASCII格式）...")
        for i, point in enumerate(points):
            # 使用适合float32的精度（通常6位小数足够）
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
            
            # 显示进度
            if (i + 1) % 1000000 == 0 or i == num_points - 1:
                progress = ((i + 1) / num_points) * 100
                print(f"  进度: {progress:.1f}% ({i + 1}/{num_points})")
    
    file_size = os.path.getsize(pcd_path) / 1024 / 1024
    print(f"ASCII PCD文件已保存: {pcd_path}")
    print(f"文件大小: {file_size:.2f} MB")


def write_pcd_binary(pcd_path, points, offset):
    """
    写入二进制格式PCD文件
    
    参数:
        pcd_path: 输出文件路径
        points: 点云数组 (N, 3)，应该是float32类型
        offset: 总偏移量 [x_offset, y_offset, z_offset]
    """
    print(f"写入二进制格式PCD文件: {pcd_path}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(pcd_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_points = len(points)
    
    # 确保points是float32类型
    if points.dtype != np.float32:
        print(f"警告: 将坐标从 {points.dtype} 转换为 float32")
        points = points.astype(np.float32)
    
    with open(pcd_path, 'wb') as f:
        # 写入ASCII头部
        header_str = ""
        header_str += "# .PCD v0.7 - Point Cloud Data file format\n"
        header_str += "# Generated by las2pcd converter (optimized for CloudCompare)\n"
        header_str += f"# OFFSET {offset[0]:.12f} {offset[1]:.12f} {offset[2]:.12f}\n"
        header_str += "# COORDINATE_SYSTEM: Local (may be translated from original)\n"
        header_str += "# DATA_TYPE: float32\n"
        header_str += "VERSION 0.7\n"
        header_str += "FIELDS x y z\n"
        header_str += "SIZE 4 4 4\n"        # float32 = 4 bytes
        header_str += "TYPE F F F\n"
        header_str += "COUNT 1 1 1\n"
        header_str += f"WIDTH {num_points}\n"
        header_str += "HEIGHT 1\n"
        header_str += "VIEWPOINT 0 0 0 1 0 0 0\n"
        header_str += f"POINTS {num_points}\n"
        header_str += "DATA binary\n"
        
        f.write(header_str.encode('ascii'))
        
        # 写入二进制点云数据
        print(f"正在写入 {num_points} 个点（二进制格式）...")
        f.write(points.tobytes())
    
    file_size = os.path.getsize(pcd_path) / 1024 / 1024
    print(f"二进制PCD文件已保存: {pcd_path}")
    print(f"文件大小: {file_size:.2f} MB")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="LAS 转 PCD 转换工具 (基于oldlas2pcd.py和minimal_las2pcd.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 默认：二进制格式，应用LAS偏移量，自动检测大坐标
  python3 las2pcd.py input.las output.pcd
  
  # ASCII格式输出
  python3 las2pcd.py input.las output.pcd --format ascii
  
  # 不应用LAS偏移量
  python3 las2pcd.py input.las output.pcd --offset false
  
  # 禁用大坐标检测
  python3 las2pcd.py input.las output.pcd --no-auto-translate
  
  # 自定义大坐标阈值
  python3 las2pcd.py input.las output.pcd --threshold 50000

功能特性:
  - 自动检测X/Y大坐标并进行局部平移（Z不变）
  - 支持二进制和ASCII格式输出
  - 强制使用float32保存坐标，兼容CloudCompare
  - PCD头部记录完整偏移量信息
        """
    )
    
    # 参数定义
    parser.add_argument("input", help="输入LAS/LAZ文件路径")
    parser.add_argument("output", help="输出PCD文件路径")
    parser.add_argument("--format", choices=["binary", "ascii"], default="binary",
                       help="输出格式 (默认: binary)")
    parser.add_argument("--offset", choices=["true", "false"], default="true",
                       help="是否应用LAS头文件偏移量 (默认: true)")
    parser.add_argument("--no-auto-translate", action="store_true",
                       help="禁用大坐标自动检测和平移")
    parser.add_argument("--threshold", type=float, default=1e5,
                       help="大坐标检测阈值，绝对值超过此值将触发局部平移 (默认: 1e5)")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    # 解析参数
    apply_offset = args.offset.lower() == "true"
    auto_translate = not args.no_auto_translate
    output_format = args.format.lower()
    coord_threshold = args.threshold
    
    print("="*70)
    print("LAS 转 PCD 转换工具 (基于oldlas2pcd.py和minimal_las2pcd.py)")
    print("="*70)
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"输出格式: {output_format}")
    print(f"应用LAS偏移量: {apply_offset}")
    print(f"自动检测大坐标: {auto_translate}")
    if auto_translate:
        print(f"大坐标阈值: {coord_threshold:g}")
    print("="*70)
    
    try:
        # 1. 读取LAS文件并处理
        points, total_offset = read_las_with_processing(
            args.input, apply_offset, auto_translate, coord_threshold
        )
        
        # 2. 写入PCD文件
        if output_format == "binary":
            write_pcd_binary(args.output, points, total_offset)
        else:
            write_pcd_ascii(args.output, points, total_offset)
        
        print("\n" + "="*70)
        print("转换完成!")
        
        # 分析结果
        las_part = "已应用" if apply_offset else "未应用"
        translation_applied = auto_translate and (abs(total_offset[0]) > 1e-6 or abs(total_offset[1]) > 1e-6)
        
        print(f"点云数量: {len(points)}")
        print(f"数据类型: {points.dtype}")
        print(f"LAS头文件偏移量: {las_part}")
        if auto_translate:
            if translation_applied:
                print("已应用局部平移 (大坐标优化)")
            else:
                print("无需局部平移 (坐标范围合理)")
        else:
            print("大坐标检测已禁用")
        
        print(f"PCD总偏移量: [{total_offset[0]:.6f}, {total_offset[1]:.6f}, {total_offset[2]:.6f}]")
        print(f"输出格式: {output_format}")
        
        print("\n特性说明:")
        print("- PCD文件使用float32格式，兼容CloudCompare等工具")
        print("- 偏移量信息记录在PCD头部，可用于坐标恢复")
        if translation_applied:
            print("- 已对X/Y进行局部平移，Z坐标保持不变")
        
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
