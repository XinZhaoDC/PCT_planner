# PCT Planner

## Overview

This is an implementation of paper **Efficient Global Navigational Planning in 3D Structures based on Point Cloud Tomography** (submitted to TMECH).
It provides a highly efficient and extensible global navigation framework based on a tomographic understanding of the environment to navigate ground robots in multi-layer structures.

**Demonstrations**: [pct_planner](https://byangw.github.io/projects/tmech2024/)

![demo](rsc/docs/demo.png)

## Prerequisites

### Environment

- Ubuntu >= 20.04
- ROS >= Noetic with ros-desktop-full installation
- CUDA >= 11.7

### Python

- Python >= 3.8
- [CuPy](https://docs.cupy.dev/en/stable/install.html) with CUDA >= 11.7
- Open3d

## Build & Install

Inside the package, there are two modules: the point cloud tomography module for tomogram reconstruction (in **tomography/**) and the planner module for path planning and optimization (in **planner/**).
You only need to build the planner module before use.
In **planner/**, run **build_thirdparty.sh** first and then run **build.sh**. 

```bash
cd planner/
./build_thirdparty.sh
./build.sh
```

## Run Examples

Three example scenarios are provided: **"Spiral"**, **"Building"**, and **"Plaza"**.
- **"Spiral"**: A spiral overpass scenario released in the [3D2M planner](https://github.com/ZJU-FAST-Lab/3D2M-planner).
- **"Building"**: A multi-layer indoor scenario with various stairs, slopes, overhangs and obstacles.
- **"Plaza"**: A complex outdoor plaza for repeated trajectory generation evaluation.

### Tomogram Construction

To plan in a scenario, first you need to construct the scene tomogram using the pcd file.
- Unzip the pcd files in **rsc/pcd/pcd_files.zip** to **rsc/pcd/**.
- For scene **"Spiral"**, you can download the pcd file from [3D2M planner spiral0.3_2.pcd](https://github.com/ZJU-FAST-Lab/3D2M-planner/tree/main/planner/src/read_pcd/PCDFiles).
- Run **roscore**, start **RViz** with the provided config (**rsc/rviz/pct_ros.rviz**). 
- In **tomography/scripts/**, run **tomography.py** with the **--scene** argument:

```bash
cd tomography/scripts/
python3 tomography.py --scene Spiral
```

- The generated tomogram is visualized as ROS PointCloud2 message in RViz and saved in **rsc/tomogram/**.

### Trajectory Generation 

After the tomogram is constructed, you can run the trajectory generation example.
- In **planner/scripts/**, run **plan.py** with the **--scene** argument:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/YOUR/DIRECTORY/TO/PCT_planner/planner/lib/3rdparty/gtsam-4.1.1/install/lib
cd planner/scripts/
python3 plan.py --scene Spiral
```

- The generated trajectory is visualized as ROS Path message in RViz.

## License

The source code is released under [GPLv2](http://www.gnu.org/licenses/) license.

For commercial use, please contact Bowen Yang [byangar@connect.ust.hk](mailto:byangar@connect.ust.hk).


# config
## tomography
__init__.py: add scene
config:add scene param
tomography.py: add help definition

## planner



# use
cd tomography/scripts/ && python3 tomography.py --scene ShenkanMLS
cd planner/scripts/ && python3 plan.py --scene ShenkanMLS
rviz -d rsc/rviz/pct_ros.rviz

## WHU-TLS batch workflow in this container

The validated runtime keeps ROS Noetic's Python 3.8 and installs the PCT Python/CUDA dependencies under the workspace instead of changing the container Python:

```text
/home/workspace/code/pct_planner/.runtime
```

The environment contains NumPy 1.24.4, CuPy CUDA 11.x 12.3.0, CUDA 11.8 user-space runtime libraries, and the Python 3.8 native planner modules. The downloaded wheels are cached in:

```text
/home/workspace/download/pct_planner_py38
```

Source the prepared environment before every PCT command:

```bash
source /opt/ros/noetic/setup.bash
source /home/workspace/code/pct_planner/src/PCT_planner/AI_prompt/environment/env_pct_planner.sh
```

Generate the WHU tomogram without RViz or a blocking ROS spin:

```bash
cd /home/workspace/code/pct_planner/src/PCT_planner/tomography/scripts
python3 tomography.py \
  --scene WHUTLSForest \
  --pcd-file /home/workspace/data/path_planning/scene/WHU_TLS_Forest.pcd \
  --output-dir /home/workspace/work/pct_planner \
  --benchmark-repeats 10 \
  --no-publish \
  --no-spin
```

Generate all five routes and both Marsim-compatible PCD and TXT files:

```bash
cd /home/workspace/code/pct_planner/src/PCT_planner/planner/scripts
python3 plan.py \
  --scene WHUTLSForest \
  --route all \
  --tomogram-dir /home/workspace/work/pct_planner \
  --output-dir /home/workspace/work/pct_planner \
  --write-pcd \
  --overwrite \
  --no-spin
```

`--route` also accepts a single value from 1 to 5. The original repository resource paths remain the defaults when explicit paths are omitted.

The structured PCD loader directly supports ASCII and binary PCD input. Open3D is retained as a fallback for other encodings, but the WHU binary input does not require Open3D visualization dependencies.

The current waypoint writer adds 1.6 m to the optimized height. Existing WHU Route 2-5 waypoints under `data/path_planning/waypoint` were generated with an effective 1.0 m addition, so newly generated Z values are exactly 0.6 m higher while XY and point counts match. Do not silently change this parameter when comparing experiments; record the selected height convention with the output.

Environment activation details, artifact classification, and the exact WHU comparison conclusion are documented in [`AI_prompt/README.md`](AI_prompt/README.md).
