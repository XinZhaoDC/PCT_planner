#!/usr/bin/env bash

PCT_WORKSPACE=/home/workspace/code/pct_planner
PCT_REPOSITORY=${PCT_WORKSPACE}/src/PCT_planner
PCT_RUNTIME=${PCT_WORKSPACE}/.runtime
PCT_SITE_PACKAGES=${PCT_RUNTIME}/python3.8/site-packages
PCT_NATIVE_BUILD=${PCT_RUNTIME}/native-build

export PYTHONPATH=${PCT_RUNTIME}/native-python:${PCT_RUNTIME}/native-python/lib:${PCT_SITE_PACKAGES}:${PYTHONPATH}
export LD_LIBRARY_PATH=${PCT_REPOSITORY}/planner/lib/3rdparty/gtsam-4.1.1/install/lib:${PCT_NATIVE_BUILD}/src/a_star:${PCT_NATIVE_BUILD}/src/common/smoothing:${PCT_NATIVE_BUILD}/src/ele_planner:${PCT_NATIVE_BUILD}/src/map_manager:${PCT_NATIVE_BUILD}/src/trajectory_optimization:${PCT_SITE_PACKAGES}/nvidia/cuda_runtime/lib:${PCT_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${PCT_RUNTIME}/cuda-libs/cublas/nvidia/cublas/lib:${PCT_RUNTIME}/cuda-libs/cufft/nvidia/cufft/lib:${PCT_RUNTIME}/cuda-libs/curand/nvidia/curand/lib:${PCT_RUNTIME}/cuda-libs/cusolver/nvidia/cusolver/lib:${PCT_RUNTIME}/cuda-libs/cusparse/nvidia/cusparse/lib:${LD_LIBRARY_PATH}
export CUPY_CACHE_DIR=${PCT_RUNTIME}/cupy-cache

unset PCT_WORKSPACE PCT_REPOSITORY PCT_RUNTIME PCT_SITE_PACKAGES PCT_NATIVE_BUILD
