"""
PCT 路径规划主程序

功能：
- 提供命令行接口运行路径规划任务
- 支持多种预定义场景的路径规划
- 集成 ROS 发布机制，实时发布规划结果
- 自动保存路径点文件供无人机使用

支持的场景：
- spiral: 螺旋形障碍物场景
- building: 建筑物环境
- plaza: 广场开放环境
- whu_tls_forest: 武汉大学激光雷达森林数据
- shenkan_mls: 沈勘移动激光扫描建筑数据
- gx_forest_gg2 / gx_forest_gg3: 广西森林环境数据

使用方法：
python plan.py --scene building
python plan.py --scene whu_tls_forest --route all

输出：
- ROS 话题: /pct_path (nav_msgs/Path)
- 文件输出: /rsc/waypoint/newwaypoint.txt

依赖：
- ROS (rospy, nav_msgs)
- NumPy
- 自定义模块 (planner_wrapper, utils, config)
"""

import argparse
import sys
from pathlib import Path as FilePath
import numpy as np

SCRIPT_DIR = FilePath(__file__).resolve().parent
PLANNER_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = PLANNER_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(PLANNER_ROOT))

import rospy
from nav_msgs.msg import Path

from utils import *
from planner_wrapper import TomogramPlanner
from config import Config
from pct_config import available_scenes, load_scene, select_routes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scene', default='spiral',
        help='Lowercase scene name. Available: ' + ', '.join(available_scenes()),
    )
    parser.add_argument(
        '--route',
        help='Route name from scene_<name>.py, or all. Omit for the scene default.',
    )
    parser.add_argument('--tomogram-file', help='Override the scene tomogram input path')
    parser.add_argument('--tomogram-dir',
                        help='Compatibility option: override only the tomogram directory')
    parser.add_argument('--output-dir',
                        help='Override the scene waypoint output directory')
    parser.add_argument('--write-pcd', action='store_true',
                        help='Also write binary PCL PointXYZ files for Marsim')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace an existing TXT output instead of appending')
    parser.add_argument('--no-spin', action='store_true',
                        help='Exit after planning instead of waiting for RViz clients')
    return parser.parse_args()


def main():
    args = parse_args()
    scene_cfg = load_scene(args.scene)
    jobs = select_routes(scene_cfg, args.route)

    if args.tomogram_file:
        tomogram_path = FilePath(args.tomogram_file).expanduser().resolve()
    elif args.tomogram_dir:
        tomogram_path = (
            FilePath(args.tomogram_dir).expanduser().resolve()
            / scene_cfg.tomogram_file.name
        )
    else:
        tomogram_path = scene_cfg.tomogram_file
    waypoint_dir = (
        FilePath(args.output_dir).expanduser().resolve()
        if args.output_dir
        else scene_cfg.waypoint_dir
    )

    rospy.init_node('pct_planner', anonymous=True)
    path_pub = rospy.Publisher('/pct_path', Path, latch=True, queue_size=1)
    planner = TomogramPlanner(
        Config(),
        waypoint_dir=waypoint_dir,
        waypoint_offset=scene_cfg.waypoint_offset,
    )
    planner.loadTomogram(tomogram_path)

    failures = []
    for route_name, route in jobs:
        start_pos = np.asarray(route.start, dtype=np.float32)
        end_pos = np.asarray(route.end, dtype=np.float32)
        traj_3d = planner.plan(start_pos, end_pos)
        if traj_3d is None:
            failures.append(route_name)
            print(f'Path planning failed: {route_name}')
            continue

        txt_path = planner.writeWaypoint(
            route.waypoint_file,
            traj_3d,
            append=not args.overwrite,
            write_pcd=args.write_pcd,
        )
        path_pub.publish(traj2ros(traj_3d))
        print(f'Waypoints saved to: {txt_path}')
        print(f'Trajectory points: {len(traj_3d)}')

    if failures:
        raise RuntimeError('Planning failed for: ' + ', '.join(failures))
    if not args.no_spin:
        rospy.spin()


if __name__ == '__main__':
    main()
