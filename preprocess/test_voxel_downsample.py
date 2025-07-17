#!/usr/bin/env python3
"""
测试体素下采样功能
================

测试修复后的体素下采样功能，验证内存优化效果
"""

import os
import sys
import numpy as np
import gc
import subprocess
from las2pcd import downsample_voxel, downsample_voxel_chunked, downsample_voxel_hash


def get_memory_usage():
    """获取当前内存使用情况（MB）"""
    try:
        # 使用系统命令获取内存使用情况
        result = subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            # RSS is in KB, convert to MB
            return float(result.stdout.strip()) / 1024
        else:
            return 0.0
    except:
        return 0.0


def create_test_points(num_points=1000000):
    """创建测试点云数据"""
    print(f"创建测试点云数据: {num_points} 个点")
    
    # 创建随机点云
    points = np.random.rand(num_points, 3) * 100  # 100x100x100的区域
    
    print(f"测试点云创建完成")
    print(f"坐标范围:")
    print(f"  X: {points[:, 0].min():.3f} - {points[:, 0].max():.3f}")
    print(f"  Y: {points[:, 1].min():.3f} - {points[:, 1].max():.3f}")
    print(f"  Z: {points[:, 2].min():.3f} - {points[:, 2].max():.3f}")
    
    return points


def test_voxel_downsampling():
    """测试体素下采样功能"""
    print("="*60)
    print("测试体素下采样功能")
    print("="*60)
    
    # 测试不同大小的点云
    test_sizes = [10000, 100000, 500000, 1000000]
    voxel_sizes = [0.5, 1.0, 2.0]
    
    for num_points in test_sizes:
        print(f"\n--- 测试点云大小: {num_points} 个点 ---")
        
        # 创建测试数据
        initial_memory = get_memory_usage()
        print(f"初始内存使用: {initial_memory:.2f} MB")
        
        points = create_test_points(num_points)
        after_creation_memory = get_memory_usage()
        print(f"创建后内存使用: {after_creation_memory:.2f} MB")
        
        for voxel_size in voxel_sizes:
            print(f"\n测试体素大小: {voxel_size}")
            
            try:
                # 测试体素下采样
                before_memory = get_memory_usage()
                print(f"下采样前内存使用: {before_memory:.2f} MB")
                
                result = downsample_voxel(points, voxel_size)
                
                after_memory = get_memory_usage()
                print(f"下采样后内存使用: {after_memory:.2f} MB")
                print(f"内存变化: {after_memory - before_memory:.2f} MB")
                
                # 验证结果
                if len(result) > 0:
                    print(f"结果验证:")
                    print(f"  输入点数: {len(points)}")
                    print(f"  输出点数: {len(result)}")
                    print(f"  压缩比: {len(result) / len(points) * 100:.1f}%")
                    print(f"  结果范围:")
                    print(f"    X: {result[:, 0].min():.3f} - {result[:, 0].max():.3f}")
                    print(f"    Y: {result[:, 1].min():.3f} - {result[:, 1].max():.3f}")
                    print(f"    Z: {result[:, 2].min():.3f} - {result[:, 2].max():.3f}")
                    print("  ✓ 下采样成功")
                else:
                    print("  ✗ 下采样失败：结果为空")
                
                # 清理内存
                del result
                gc.collect()
                
            except Exception as e:
                print(f"  ✗ 下采样失败: {e}")
                
                # 尝试分块处理
                try:
                    print("  尝试分块处理...")
                    result = downsample_voxel_chunked(points, voxel_size)
                    print(f"  分块处理成功: {len(points)} -> {len(result)} 点")
                    del result
                    gc.collect()
                except Exception as e2:
                    print(f"  分块处理也失败: {e2}")
                    
                    # 尝试哈希方法
                    try:
                        print("  尝试哈希方法...")
                        result = downsample_voxel_hash(points, voxel_size)
                        print(f"  哈希方法成功: {len(points)} -> {len(result)} 点")
                        del result
                        gc.collect()
                    except Exception as e3:
                        print(f"  哈希方法也失败: {e3}")
        
        # 清理测试数据
        del points
        gc.collect()
        
        final_memory = get_memory_usage()
        print(f"测试完成后内存使用: {final_memory:.2f} MB")


def test_specific_case():
    """测试特定的问题案例"""
    print("\n" + "="*60)
    print("测试特定问题案例")
    print("="*60)
    
    # 模拟真实LAS文件的情况
    print("创建模拟真实LAS数据...")
    
    # 创建一个较大的点云，包含一些聚集的点
    num_points = 2000000  # 200万个点
    
    # 创建三个聚集区域
    cluster1 = np.random.normal([10, 10, 5], [2, 2, 1], (num_points//3, 3))
    cluster2 = np.random.normal([50, 50, 10], [3, 3, 2], (num_points//3, 3))
    cluster3 = np.random.normal([80, 80, 15], [1, 1, 0.5], (num_points//3, 3))
    
    points = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"创建的点云:")
    print(f"  点数: {len(points)}")
    print(f"  内存使用: {points.nbytes / 1024 / 1024:.2f} MB")
    print(f"  坐标范围:")
    print(f"    X: {points[:, 0].min():.3f} - {points[:, 0].max():.3f}")
    print(f"    Y: {points[:, 1].min():.3f} - {points[:, 1].max():.3f}")
    print(f"    Z: {points[:, 2].min():.3f} - {points[:, 2].max():.3f}")
    
    # 测试不同的体素大小
    test_voxel_sizes = [0.1, 0.5, 1.0, 2.0]
    
    for voxel_size in test_voxel_sizes:
        print(f"\n--- 测试体素大小: {voxel_size} ---")
        
        try:
            before_memory = get_memory_usage()
            print(f"开始前内存: {before_memory:.2f} MB")
            
            result = downsample_voxel(points, voxel_size)
            
            after_memory = get_memory_usage()
            print(f"完成后内存: {after_memory:.2f} MB")
            print(f"内存峰值变化: {after_memory - before_memory:.2f} MB")
            
            print(f"下采样结果:")
            print(f"  {len(points)} -> {len(result)} 点")
            print(f"  压缩比: {len(result) / len(points) * 100:.2f}%")
            
            # 验证结果合理性
            if len(result) > 0:
                expected_voxels = ((points[:, 0].max() - points[:, 0].min()) / voxel_size) * \
                                ((points[:, 1].max() - points[:, 1].min()) / voxel_size) * \
                                ((points[:, 2].max() - points[:, 2].min()) / voxel_size)
                print(f"  理论最大体素数: {expected_voxels:.0f}")
                print(f"  实际体素数: {len(result)}")
                print(f"  体素填充率: {len(result) / expected_voxels * 100:.2f}%")
                print("  ✓ 测试通过")
            else:
                print("  ✗ 结果为空")
            
            del result
            gc.collect()
            
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            print(f"  错误类型: {type(e).__name__}")
            
            # 尝试分块处理
            try:
                print("  尝试分块处理...")
                result = downsample_voxel_chunked(points, voxel_size, chunk_size=50000)
                print(f"  分块处理成功: {len(points)} -> {len(result)} 点")
                del result
                gc.collect()
            except Exception as e2:
                print(f"  分块处理失败: {e2}")
    
    print("\n特定案例测试完成")


def main():
    """主函数"""
    print("体素下采样功能测试")
    print("="*60)
    
    # 显示系统信息
    print(f"系统信息:")
    try:
        # 获取内存信息
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if 'MemTotal:' in line:
                    total_mem = int(line.split()[1]) // 1024 // 1024  # Convert to GB
                    print(f"  总内存: {total_mem:.2f} GB")
                elif 'MemAvailable:' in line:
                    avail_mem = int(line.split()[1]) // 1024 // 1024  # Convert to GB
                    print(f"  可用内存: {avail_mem:.2f} GB")
    except:
        print("  无法获取内存信息")
    
    try:
        # 获取CPU信息
        cpu_count = os.cpu_count()
        print(f"  CPU核心数: {cpu_count}")
    except:
        print("  无法获取CPU信息")
    
    try:
        # 基础功能测试
        test_voxel_downsampling()
        
        # 特定案例测试
        test_specific_case()
        
        print("\n" + "="*60)
        print("所有测试完成")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
