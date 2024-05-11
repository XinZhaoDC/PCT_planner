import sys
import argparse
import numpy as np

import rospy
from nav_msgs.msg import Path

from utils import *
from planner_wrapper import TomogramPlanner

sys.path.append('../')
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default='Spiral', help='Name of the scene. Available: [\'Spiral\', \'Building\', \'Plaza\', \'Forest\', \'ShenkanMLS\']')
args = parser.parse_args()

cfg = Config()

if args.scene == 'Spiral':
    tomo_file = 'spiral0.3_2'
    start_pos = np.array([-16.0, -6.0], dtype=np.float32)
    end_pos = np.array([-26.0, -5.0], dtype=np.float32)
elif args.scene == 'Building':
    tomo_file = 'building2_9'
    start_pos = np.array([5.0, 5.0], dtype=np.float32)
    end_pos = np.array([-6.0, -1.0], dtype=np.float32)
elif args.scene == 'Forest':
    tomo_file = 'WHU_TLS_Forest'
    start_pos = np.array([13.40,19.27], dtype=np.float32)
    end_pos = np.array([43.05,60.35], dtype=np.float32)
elif args.scene == 'ShenkanMLS':
    tomo_file = 'shenkan_MLS_building'
    start_pos = np.array([507, 115], dtype=np.float32)
    end_pos = np.array([54, 20], dtype=np.float32)
else:
    tomo_file = 'plaza3_10'
    start_pos = np.array([0.0, 0.0], dtype=np.float32)
    end_pos = np.array([23.0, 10.0], dtype=np.float32)
    
'''
start pose and end pose for Forest:
[21.22,7.34] [14.93,62.67]
[35.96,6.53] [28.89,67.36]
[45.33,16.55] [36.53,67.18]
[13.40,19.27] [43.05,60.35]
[9.19,26.52] [48.18,54.52]
'''

'''
start pose and end pose for Shenkan_MLS:
[507.9080 115.0592 1.4566] [54.6010 20.4289 1.4659]
'''

waypoint_file = 'waypoint'
path_pub = rospy.Publisher("/pct_path", Path, latch=True, queue_size=1)
planner = TomogramPlanner(cfg)

def pct_plan():
    planner.loadTomogram(tomo_file)

    traj_3d = planner.plan(start_pos, end_pos)
    if traj_3d is not None:
        planner.writeWaypoint(waypoint_file,traj_3d)
        path_pub.publish(traj2ros(traj_3d))
        print("Trajectory published")


if __name__ == '__main__':
    rospy.init_node("pct_planner", anonymous=True)

    pct_plan()

    rospy.spin()