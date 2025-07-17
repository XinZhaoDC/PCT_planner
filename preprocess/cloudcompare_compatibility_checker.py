#!/usr/bin/env python3
"""
CloudCompare兼容性检查器
=======================

检查PCD文件格式是否符合CloudCompare的解析要求，分析可能导致显示异常的原因。

功能：
1. 验证PCD文件头部格式
2. 检查字段定义和数据类型
3. 分析坐标数值范围和精度
4. 检测可能的解析错误
5. 提供修复建议
"""

import os
import sys
import re
import numpy as np
from typing import Dict, List, Tuple, Optional


def check_pcd_header(pcd_file: str) -> Dict:
    """
    检查PCD文件头部格式
    
    参数:
        pcd_file: PCD文件路径
    
    返回:
        检查结果字典
    """
    print(f"检查PCD文件头部: {pcd_file}")
    
    if not os.path.exists(pcd_file):
        return {"error": f"文件不存在: {pcd_file}"}
    
    header_info = {}
    issues = []
    
    try:
        with open(pcd_file, 'r', encoding='utf-8') as f:
            lines = []
            line_count = 0
            
            # 读取头部信息
            for line in f:
                line_count += 1
                line = line.strip()
                lines.append(line)
                
                # 如果遇到DATA行，头部结束
                if line.startswith('DATA'):
                    break
                
                # 防止读取过多行
                if line_count > 50:
                    break
        
        # 解析头部字段
        version_found = False
        fields_found = False
        size_found = False
        type_found = False
        count_found = False
        width_found = False
        height_found = False
        points_found = False
        data_found = False
        
        for line in lines:
            if line.startswith('VERSION'):
                version_found = True
                header_info['version'] = line.split()[1] if len(line.split()) > 1 else None
            elif line.startswith('FIELDS'):
                fields_found = True
                header_info['fields'] = line.split()[1:]
            elif line.startswith('SIZE'):
                size_found = True
                header_info['sizes'] = [int(x) for x in line.split()[1:]]
            elif line.startswith('TYPE'):
                type_found = True
                header_info['types'] = line.split()[1:]
            elif line.startswith('COUNT'):
                count_found = True
                header_info['counts'] = [int(x) for x in line.split()[1:]]
            elif line.startswith('WIDTH'):
                width_found = True
                header_info['width'] = int(line.split()[1])
            elif line.startswith('HEIGHT'):
                height_found = True
                header_info['height'] = int(line.split()[1])
            elif line.startswith('POINTS'):
                points_found = True
                header_info['points'] = int(line.split()[1])
            elif line.startswith('DATA'):
                data_found = True
                header_info['data_format'] = line.split()[1] if len(line.split()) > 1 else None
        
        # 检查必需字段
        required_fields = {
            'VERSION': version_found,
            'FIELDS': fields_found,
            'SIZE': size_found,
            'TYPE': type_found,
            'COUNT': count_found,
            'WIDTH': width_found,
            'HEIGHT': height_found,
            'POINTS': points_found,
            'DATA': data_found
        }
        
        for field, found in required_fields.items():
            if not found:
                issues.append(f"缺少必需字段: {field}")
        
        # 检查字段一致性
        if fields_found and size_found and type_found and count_found:
            field_count = len(header_info['fields'])
            if len(header_info['sizes']) != field_count:
                issues.append(f"SIZE字段数量({len(header_info['sizes'])})与FIELDS不匹配({field_count})")
            if len(header_info['types']) != field_count:
                issues.append(f"TYPE字段数量({len(header_info['types'])})与FIELDS不匹配({field_count})")
            if len(header_info['counts']) != field_count:
                issues.append(f"COUNT字段数量({len(header_info['counts'])})与FIELDS不匹配({field_count})")
        
        # 检查点数一致性
        if width_found and height_found and points_found:
            calculated_points = header_info['width'] * header_info['height']
            if calculated_points != header_info['points']:
                issues.append(f"POINTS({header_info['points']})与WIDTH*HEIGHT({calculated_points})不匹配")
        
        # 检查数据格式
        if data_found and header_info['data_format'] not in ['ascii', 'binary']:
            issues.append(f"不支持的数据格式: {header_info['data_format']}")
        
        header_info['issues'] = issues
        header_info['valid'] = len(issues) == 0
        
        return header_info
        
    except Exception as e:
        return {"error": f"读取文件时出错: {e}"}


def check_coordinate_precision(pcd_file: str, sample_size: int = 10000) -> Dict:
    """
    检查坐标精度和数值范围
    
    参数:
        pcd_file: PCD文件路径
        sample_size: 采样点数
    
    返回:
        精度检查结果
    """
    print(f"检查坐标精度: {pcd_file}")
    
    precision_info = {}
    issues = []
    
    try:
        # 读取头部信息
        header_result = check_pcd_header(pcd_file)
        if 'error' in header_result:
            return header_result
        
        data_format = header_result.get('data_format', 'ascii')
        
        if data_format == 'ascii':
            # ASCII格式
            with open(pcd_file, 'r', encoding='utf-8') as f:
                # 跳过头部
                for line in f:
                    if line.strip().startswith('DATA'):
                        break
                
                # 读取数据点
                points = []
                count = 0
                for line in f:
                    if count >= sample_size:
                        break
                    
                    line = line.strip()
                    if line:
                        try:
                            coords = [float(x) for x in line.split()]
                            if len(coords) >= 3:
                                points.append(coords[:3])
                                count += 1
                        except ValueError:
                            issues.append(f"无效的坐标数据: {line}")
                
                if points:
                    points = np.array(points)
                    
                    # 分析坐标范围
                    precision_info['sample_count'] = len(points)
                    precision_info['x_range'] = [float(points[:, 0].min()), float(points[:, 0].max())]
                    precision_info['y_range'] = [float(points[:, 1].min()), float(points[:, 1].max())]
                    precision_info['z_range'] = [float(points[:, 2].min()), float(points[:, 2].max())]
                    
                    # 检查数值范围是否合理
                    x_span = precision_info['x_range'][1] - precision_info['x_range'][0]
                    y_span = precision_info['y_range'][1] - precision_info['y_range'][0]
                    z_span = precision_info['z_range'][1] - precision_info['z_range'][0]
                    
                    precision_info['x_span'] = x_span
                    precision_info['y_span'] = y_span
                    precision_info['z_span'] = z_span
                    
                    # 检查异常大的坐标范围
                    if x_span > 1e6 or y_span > 1e6 or z_span > 1e6:
                        issues.append(f"坐标范围过大: X={x_span:.3f}, Y={y_span:.3f}, Z={z_span:.3f}")
                    
                    # 检查异常小的坐标范围
                    if x_span < 1e-6 or y_span < 1e-6 or z_span < 1e-6:
                        issues.append(f"坐标范围过小: X={x_span:.3f}, Y={y_span:.3f}, Z={z_span:.3f}")
                    
                    # 检查坐标精度
                    x_precision = estimate_precision(points[:, 0])
                    y_precision = estimate_precision(points[:, 1])
                    z_precision = estimate_precision(points[:, 2])
                    
                    precision_info['x_precision'] = x_precision
                    precision_info['y_precision'] = y_precision
                    precision_info['z_precision'] = z_precision
                    
                    # 检查是否存在NaN或Inf
                    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                        issues.append("坐标数据中存在NaN或Inf值")
                    
                    # 检查是否所有点都在同一位置
                    if x_span == 0 and y_span == 0 and z_span == 0:
                        issues.append("所有点都在同一位置")
                
        else:
            # 二进制格式
            issues.append("二进制格式检查尚未实现")
        
        precision_info['issues'] = issues
        precision_info['valid'] = len(issues) == 0
        
        return precision_info
        
    except Exception as e:
        return {"error": f"检查精度时出错: {e}"}


def estimate_precision(values: np.ndarray) -> float:
    """
    估算数值精度
    
    参数:
        values: 数值数组
    
    返回:
        估算的精度值
    """
    if len(values) < 2:
        return 0.0
    
    # 计算相邻值的最小差异
    sorted_values = np.sort(np.unique(values))
    if len(sorted_values) < 2:
        return 0.0
    
    diffs = np.diff(sorted_values)
    min_diff = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else 0.0
    
    return min_diff


def check_cloudcompare_compatibility(pcd_file: str) -> Dict:
    """
    检查CloudCompare兼容性
    
    参数:
        pcd_file: PCD文件路径
    
    返回:
        兼容性检查结果
    """
    print(f"检查CloudCompare兼容性: {pcd_file}")
    
    compatibility_info = {}
    issues = []
    warnings = []
    
    # 1. 检查头部格式
    header_result = check_pcd_header(pcd_file)
    if 'error' in header_result:
        return header_result
    
    compatibility_info['header'] = header_result
    issues.extend(header_result.get('issues', []))
    
    # 2. 检查坐标精度
    precision_result = check_coordinate_precision(pcd_file)
    if 'error' in precision_result:
        return precision_result
    
    compatibility_info['precision'] = precision_result
    issues.extend(precision_result.get('issues', []))
    
    # 3. CloudCompare特定检查
    
    # 检查字段名称
    fields = header_result.get('fields', [])
    if not all(field in ['x', 'y', 'z'] for field in fields):
        warnings.append(f"包含CloudCompare可能不支持的字段: {fields}")
    
    # 检查数据类型
    types = header_result.get('types', [])
    if not all(t in ['F', 'I', 'U'] for t in types):
        warnings.append(f"包含CloudCompare可能不支持的数据类型: {types}")
    
    # 检查点数量
    points = header_result.get('points', 0)
    if points > 10000000:  # 1000万点
        warnings.append(f"点数量很大({points})，可能影响CloudCompare性能")
    
    # 检查坐标范围
    if 'precision' in compatibility_info:
        x_range = compatibility_info['precision'].get('x_range', [0, 0])
        y_range = compatibility_info['precision'].get('y_range', [0, 0])
        z_range = compatibility_info['precision'].get('z_range', [0, 0])
        
        # 检查坐标是否过大
        max_coord = max(abs(x_range[0]), abs(x_range[1]), 
                       abs(y_range[0]), abs(y_range[1]),
                       abs(z_range[0]), abs(z_range[1]))
        
        if max_coord > 1e6:
            warnings.append(f"坐标值过大({max_coord:.3f})，可能导致CloudCompare显示问题")
        
        # 检查坐标精度
        x_precision = compatibility_info['precision'].get('x_precision', 0)
        y_precision = compatibility_info['precision'].get('y_precision', 0)
        z_precision = compatibility_info['precision'].get('z_precision', 0)
        
        if x_precision < 1e-6 or y_precision < 1e-6 or z_precision < 1e-6:
            warnings.append("坐标精度过高，可能导致CloudCompare显示问题")
    
    # 4. 检查文件编码
    try:
        with open(pcd_file, 'rb') as f:
            first_bytes = f.read(1000)
            # 检查是否包含非ASCII字符
            try:
                first_bytes.decode('ascii')
            except UnicodeDecodeError:
                warnings.append("文件包含非ASCII字符，可能导致解析问题")
    except Exception as e:
        warnings.append(f"无法检查文件编码: {e}")
    
    compatibility_info['cloudcompare_issues'] = issues
    compatibility_info['cloudcompare_warnings'] = warnings
    compatibility_info['cloudcompare_compatible'] = len(issues) == 0
    
    return compatibility_info


def generate_cloudcompare_compatible_pcd(input_pcd: str, output_pcd: str) -> bool:
    """
    生成CloudCompare兼容的PCD文件
    
    参数:
        input_pcd: 输入PCD文件
        output_pcd: 输出PCD文件
    
    返回:
        是否成功
    """
    print(f"生成CloudCompare兼容的PCD文件: {input_pcd} -> {output_pcd}")
    
    try:
        # 检查输入文件
        compatibility_result = check_cloudcompare_compatibility(input_pcd)
        if 'error' in compatibility_result:
            print(f"错误: {compatibility_result['error']}")
            return False
        
        # 读取原始数据
        header_info = compatibility_result['header']
        data_format = header_info.get('data_format', 'ascii')
        
        if data_format != 'ascii':
            print("暂不支持二进制格式转换")
            return False
        
        # 读取点云数据
        points = []
        with open(input_pcd, 'r', encoding='utf-8') as f:
            # 跳过头部
            for line in f:
                if line.strip().startswith('DATA'):
                    break
            
            # 读取数据点
            for line in f:
                line = line.strip()
                if line:
                    try:
                        coords = [float(x) for x in line.split()]
                        if len(coords) >= 3:
                            points.append(coords[:3])
                    except ValueError:
                        continue
        
        if not points:
            print("没有有效的点云数据")
            return False
        
        points = np.array(points)
        
        # 数据清理和规范化
        # 1. 移除NaN和Inf值
        valid_mask = np.all(np.isfinite(points), axis=1)
        points = points[valid_mask]
        
        if len(points) == 0:
            print("清理后没有有效点")
            return False
        
        # 2. 坐标范围检查和调整
        x_range = [points[:, 0].min(), points[:, 0].max()]
        y_range = [points[:, 1].min(), points[:, 1].max()]
        z_range = [points[:, 2].min(), points[:, 2].max()]
        
        print(f"原始坐标范围:")
        print(f"  X: {x_range[0]:.6f} - {x_range[1]:.6f}")
        print(f"  Y: {y_range[0]:.6f} - {y_range[1]:.6f}")
        print(f"  Z: {z_range[0]:.6f} - {z_range[1]:.6f}")
        
        # 3. 如果坐标过大，进行中心化
        max_coord = max(abs(x_range[0]), abs(x_range[1]), 
                       abs(y_range[0]), abs(y_range[1]),
                       abs(z_range[0]), abs(z_range[1]))
        
        offset = np.array([0.0, 0.0, 0.0])
        if max_coord > 1e6:
            print("坐标过大，进行中心化处理...")
            centroid = np.mean(points, axis=0)
            points = points - centroid
            offset = centroid
            
            print(f"中心化偏移量: [{offset[0]:.6f}, {offset[1]:.6f}, {offset[2]:.6f}]")
        
        # 4. 限制精度以避免过高精度问题
        points = np.round(points, 6)  # 保留6位小数
        
        # 5. 写入CloudCompare兼容的PCD文件
        num_points = len(points)
        
        with open(output_pcd, 'w', encoding='utf-8') as f:
            # 写入标准头部
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("# Generated by CloudCompare compatibility converter\n")
            if not np.allclose(offset, 0):
                f.write(f"# OFFSET {offset[0]:.12f} {offset[1]:.12f} {offset[2]:.12f}\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")  # 使用float32以减少文件大小
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
            f.write(f"WIDTH {num_points}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {num_points}\n")
            f.write("DATA ascii\n")
            
            # 写入点数据
            for point in points:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        print(f"成功生成CloudCompare兼容的PCD文件: {output_pcd}")
        print(f"点数量: {num_points}")
        
        # 验证生成的文件
        verification_result = check_cloudcompare_compatibility(output_pcd)
        if verification_result.get('cloudcompare_compatible', False):
            print("验证通过: 生成的文件兼容CloudCompare")
        else:
            print("警告: 生成的文件可能仍存在兼容性问题")
            for issue in verification_result.get('cloudcompare_issues', []):
                print(f"  - {issue}")
        
        return True
        
    except Exception as e:
        print(f"生成兼容文件时出错: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CloudCompare兼容性检查器")
    parser.add_argument("input_pcd", help="输入PCD文件路径")
    parser.add_argument("--output", help="输出兼容的PCD文件路径")
    parser.add_argument("--detailed", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_pcd):
        print(f"错误: 输入文件不存在: {args.input_pcd}")
        return 1
    
    print("="*60)
    print("CloudCompare兼容性检查器")
    print("="*60)
    
    # 检查兼容性
    result = check_cloudcompare_compatibility(args.input_pcd)
    
    if 'error' in result:
        print(f"错误: {result['error']}")
        return 1
    
    # 显示结果
    print(f"文件: {args.input_pcd}")
    print(f"CloudCompare兼容性: {'✓' if result['cloudcompare_compatible'] else '✗'}")
    
    if result['cloudcompare_issues']:
        print("\n严重问题:")
        for issue in result['cloudcompare_issues']:
            print(f"  ✗ {issue}")
    
    if result['cloudcompare_warnings']:
        print("\n警告:")
        for warning in result['cloudcompare_warnings']:
            print(f"  ⚠ {warning}")
    
    if args.detailed:
        print("\n详细信息:")
        print("-" * 40)
        
        # 头部信息
        header = result.get('header', {})
        print(f"VERSION: {header.get('version', 'N/A')}")
        print(f"FIELDS: {header.get('fields', 'N/A')}")
        print(f"SIZE: {header.get('sizes', 'N/A')}")
        print(f"TYPE: {header.get('types', 'N/A')}")
        print(f"COUNT: {header.get('counts', 'N/A')}")
        print(f"WIDTH: {header.get('width', 'N/A')}")
        print(f"HEIGHT: {header.get('height', 'N/A')}")
        print(f"POINTS: {header.get('points', 'N/A')}")
        print(f"DATA: {header.get('data_format', 'N/A')}")
        
        # 精度信息
        precision = result.get('precision', {})
        if precision:
            print(f"\n坐标范围:")
            x_range = precision.get('x_range', [0, 0])
            y_range = precision.get('y_range', [0, 0])
            z_range = precision.get('z_range', [0, 0])
            print(f"  X: {x_range[0]:.6f} - {x_range[1]:.6f}")
            print(f"  Y: {y_range[0]:.6f} - {y_range[1]:.6f}")
            print(f"  Z: {z_range[0]:.6f} - {z_range[1]:.6f}")
            
            print(f"\n坐标精度:")
            print(f"  X: {precision.get('x_precision', 0):.9f}")
            print(f"  Y: {precision.get('y_precision', 0):.9f}")
            print(f"  Z: {precision.get('z_precision', 0):.9f}")
    
    # 生成兼容文件
    if args.output:
        print("\n" + "="*60)
        success = generate_cloudcompare_compatible_pcd(args.input_pcd, args.output)
        if success:
            print("✓ 成功生成CloudCompare兼容的PCD文件")
        else:
            print("✗ 生成CloudCompare兼容的PCD文件失败")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
