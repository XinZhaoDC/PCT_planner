"""
PCT 路径规划主程序

功能：
- 提供命令行接口运行路径规划任务
- 支持多种预定义场景的路径规划
- 集成 ROS 发布机制，实时发布规划结果
- 自动保存路径点文件供无人机使用

支持的场景：
- Spiral: 螺旋形障碍物场景
- Building: 建筑物环境
- Plaza: 广场开放环境
- WHUTLSForest: 武汉大学激光雷达森林数据
- ShenkanMLS: 沈勘移动激光扫描建筑数据
- GXForestGG2: 广西森林环境数据

使用方法：
python plan.py --scene Building
python plan.py --scene WHUTLSForest

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
import numpy as np

import rospy
from nav_msgs.msg import Path

from utils import *
from planner_wrapper import TomogramPlanner

sys.path.append('../')
from config import Config

SCENES = {
    'Spiral': ('spiral0.3_2', [-16.0, -6.0], [-26.0, -5.0]),
    'Building': ('building2_9', [5.0, 5.0], [-6.0, -1.0]),
    'ShenkanMLS': ('shenkan_MLS_building', [54.0, 20.0], [507.0, 115.0]),
    'GXForestGG2': ('GXForestGG2', [-41.898, 149.540], [65.612, 74.909]),
    'GXForestGG3': ('GXForestGG3', [-86.218, -11.500], [77.656, 0.500]),
    'Plaza': ('plaza3_10', [0.0, 0.0], [23.0, 10.0]),
}

WHU_ROUTES = {
    1: ([21.22, 7.34], [14.93, 62.67]),
    2: ([35.96, 6.53], [28.89, 67.36]),
    3: ([45.33, 16.55], [36.53, 67.18]),
    4: ([13.40, 19.27], [43.05, 60.35]),
    5: ([9.19, 26.52], [48.18, 54.52]),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scene', default='Spiral',
        choices=['Spiral', 'Building', 'Plaza', 'WHUTLSForest',
                 'ShenkanMLS', 'GXForestGG2', 'GXForestGG3'],
    )
    parser.add_argument(
        '--route', choices=['1', '2', '3', '4', '5', 'all'],
        help='WHUTLSForest route. Omit to preserve the original route-4 behavior.',
    )
    parser.add_argument('--tomogram-dir', help='Absolute tomogram input directory')
    parser.add_argument('--output-dir', help='Absolute waypoint output directory')
    parser.add_argument('--write-pcd', action='store_true',
                        help='Also write binary PCL PointXYZ files for Marsim')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace an existing TXT output instead of appending')
    parser.add_argument('--no-spin', action='store_true',
                        help='Exit after planning instead of waiting for RViz clients')
    return parser.parse_args()


def planning_jobs(args):
    if args.scene != 'WHUTLSForest':
        if args.route is not None:
            raise ValueError('--route is only valid for WHUTLSForest')
        tomo_file, start, end = SCENES[args.scene]
        return tomo_file, [(tomo_file + '_waypoint', start, end)]

    tomo_file = 'WHU_TLS_Forest'
    if args.route is None:
        start, end = WHU_ROUTES[4]
        return tomo_file, [('WHU_TLS_Forest_waypoint', start, end)]

    route_ids = range(1, 6) if args.route == 'all' else [int(args.route)]
    jobs = []
    for route_id in route_ids:
        start, end = WHU_ROUTES[route_id]
        jobs.append((f'waypoint_{route_id}', start, end))
    return tomo_file, jobs


def main():
    args = parse_args()
    tomo_file, jobs = planning_jobs(args)

    rospy.init_node('pct_planner', anonymous=True)
    path_pub = rospy.Publisher('/pct_path', Path, latch=True, queue_size=1)
    planner = TomogramPlanner(
        Config(), tomo_dir=args.tomogram_dir, waypoint_dir=args.output_dir
    )
    planner.loadTomogram(tomo_file)

    failures = []
    for waypoint_file, start, end in jobs:
        start_pos = np.asarray(start, dtype=np.float32)
        end_pos = np.asarray(end, dtype=np.float32)
        traj_3d = planner.plan(start_pos, end_pos)
        if traj_3d is None:
            failures.append(waypoint_file)
            print(f'Path planning failed: {waypoint_file}')
            continue

        txt_path = planner.writeWaypoint(
            waypoint_file,
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
