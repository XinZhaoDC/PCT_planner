import string
import cupy as cp


def utils_point(resolution, n_row, n_col):
    """
    点处理工具函数生成器
    
    功能：
    - 生成用于点云处理的 CUDA 设备函数
    - 提供坐标转换和索引计算的基础工具
    - 包含原子操作函数用于并行环境下的安全更新
    
    参数：
    - resolution: 网格分辨率（米/像素）
    - n_row: 网格行数
    - n_col: 网格列数
    
    注意：
    - layer_size 在函数内部计算为 n_row * n_col
    - 用于多层地图的索引计算
    
    返回的设备函数：
    - getIndexLine: 将连续坐标转换为离散网格索引
    - getIndexMap_1d: 将2D坐标转换为1D数组索引
    - getIndexBlock_1d: 获取多层地图中特定层的索引
    - atomicMaxFloat: 原子操作更新最大值
    - atomicMinFloat: 原子操作更新最小值
    """
    util_preamble = string.Template(
        '''
        // Convert continuous coordinate to discrete grid index
        __device__ int getIndexLine(float16 x, float16 center)
        {
            int i = round((x - center) / ${resolution});
            return i;
        }

        // Convert 2D coordinates to 1D array index within a layer
        __device__ int getIndexMap_1d(float16 x, float16 y, float16 cx, float16 cy)
        {
            // Return 1D index of a point (x, y) in a layer
            // Calculate grid indices with offset to center
            int idx_x = getIndexLine(x, cx) + ${n_row} / 2;
            int idx_y = getIndexLine(y, cy) + ${n_col} / 2;

            // Check if the index is inside the map
            // Boundary check - return -1 if outside map
            if (idx_x < 0 || idx_x >= ${n_row} || idx_y < 0 || idx_y >= ${n_col})
            {
                return -1;
            }
            // Convert 2D index to 1D (row-major order)
            return ${n_col} * idx_x + idx_y;
        }

        // Get 1D index for multi-layer map block
        __device__ int getIndexBlock_1d(int idx, int layer_n)
        {
            // Return 1D index of a point (x, y) in multi-layer map block
            // Calculate offset for specific layer
            return (int)${layer_size} * layer_n + idx;
        }

        // Atomic operation for maximum float value (thread-safe)
        __device__ static float atomicMaxFloat(float* address, float val)
        {
            int* address_as_i = (int*) address;
            int old = *address_as_i, assumed;
            do {
                assumed = old;
                old = ::atomicCAS(address_as_i, assumed,
                    __float_as_int(::fmaxf(val, __int_as_float(assumed))));
            } while (assumed != old);

            return __int_as_float(old);
        }

        // Atomic operation for minimum float value (thread-safe)
        __device__ static float atomicMinFloat(float* address, float val)
        {
            int* address_as_i = (int*) address;
            int old = *address_as_i, assumed;
            do {
                assumed = old;
                old = ::atomicCAS(address_as_i, assumed,
                    __float_as_int(::fminf(val, __int_as_float(assumed))));
            } while (assumed != old);

            return __int_as_float(old);
        }
        '''
    ).substitute(
        resolution=resolution,
        n_row=n_row, 
        n_col=n_col,
        layer_size=n_row*n_col
    )

    return util_preamble


def utils_map(n_row, n_col):
    """
    地图处理工具函数生成器
    
    功能：
    - 生成用于地图操作的 CUDA 设备函数
    - 提供邻域访问和相对位置计算功能
    - 支持多层地图的索引转换
    
    参数：
    - n_row: 网格行数
    - n_col: 网格列数
    
    注意：
    - layer_size 在函数内部计算为 n_row * n_col
    - 表示单层地图的总网格数量
    - 用于多层地图结构中的索引计算
    
    返回的设备函数：
    - getIdxRelative: 计算相对偏移位置的1D索引
      * 输入当前索引和相对偏移(dx, dy)
      * 返回相对位置的索引，越界时返回-1
      * 用于邻域操作如卷积、膨胀等
    """
    util_preamble=string.Template(
        '''
        // Return 1D index of the relative point (x+dx, y+dy) in multi-layer map block
        // Get 1D index of relative neighbor point (x+dx, y+dy)
        __device__ int getIdxRelative(int idx, int dx, int dy) 
        {
            // Extract 2D coordinates from 1D index
            int idx_2d = idx % (int)${layer_size};
            int idx_x = idx_2d / ${n_col};
            int idx_y = idx_2d % ${n_col};
            
            // Calculate relative position
            int idx_rx = idx_x + dx;
            int idx_ry = idx_y + dy;

            // Boundary check - return -1 if outside map bounds
            if ( idx_rx < 0 || idx_rx > (${n_row} - 1) ) 
                return -1;
            if ( idx_ry < 0 || idx_ry > (${n_col} - 1) )
                return -1;

            // Calculate relative offset and add to original index
            return ${n_col} * dx + dy + idx;
        }
        '''
    ).substitute(
        n_row=n_row, 
        n_col=n_col,
        layer_size=n_row*n_col
    )

    return util_preamble


def tomographyKernel(resolution, n_row, n_col, n_slice, slice_h0, slice_dh):
    """
    层析成像内核函数
    
    功能：
    - 处理 3D 点云数据，将其分层投影到 2D 网格上
    - 对每个网格单元计算地面高度和天花板高度
    - 使用原子操作确保并行处理时的数据一致性
    
    参数：
    - resolution: 网格分辨率
    - n_row, n_col: 网格行列数
    - n_slice: 切片数量
    - slice_h0: 起始切片高度
    - slice_dh: 切片间隔
    
    内核逻辑：
    1. 提取点云的 3D 坐标 (px, py, pz)
    2. 计算点在 2D 网格中的索引位置
    3. 对每个高度切片：
       - 如果点在切片下方，更新地面高度 (layers_g)
       - 如果点在切片上方，更新天花板高度 (layers_c)
    """
    tomography_kernel = cp.ElementwiseKernel(
        in_params='raw U points, raw U center',
        out_params='raw U layers_g, raw U layers_c',
        preamble=utils_point(resolution, n_row, n_col),
        operation=string.Template(
            '''
            // Extract 3D coordinates from point cloud
            U px = points[i * 3];
            U py = points[i * 3 + 1];
            U pz = points[i * 3 + 2];

            // Calculate 2D grid index
            int idx = getIndexMap_1d(px, py, center[0], center[1]);
            if ( idx < 0 ) 
                return; 
            
            // Process each height slice
            for ( int s_idx = 0; s_idx < ${n_slice}; s_idx ++ )
            {
                U slice = ${slice_h0} + s_idx * ${slice_dh};
                if ( pz <= slice )
                    atomicMaxFloat(&layers_g[getIndexBlock_1d(idx, s_idx)], pz);
                else
                    atomicMinFloat(&layers_c[getIndexBlock_1d(idx, s_idx)], pz);
            }
            '''
        ).substitute(
            n_slice=n_slice,
            slice_h0=slice_h0,
            slice_dh=slice_dh
        ),
        name='tomography_kernel'
    )
                            
    return tomography_kernel


def travKernel(
    n_row, n_col, half_kernel_size,
    interval_min, interval_free, step_cross, step_stand, standable_th, cost_barrier
    ):
    """
    可通行性代价计算内核
    
    功能：
    - 基于地形间隔和梯度信息计算每个网格的通行代价
    - 考虑机器人的步行能力和安全性要求
    - 区分可站立区域、可跨越区域和障碍区域
    
    参数说明：
    - interval_min: 最小可通行间隔
    - interval_free: 自由通行间隔阈值
    - step_cross: 可跨越的最大梯度
    - step_stand: 可站立的最大梯度
    - standable_th: 周围可站立网格的最少数量
    - cost_barrier: 障碍物代价值
    
    计算逻辑：
    1. 检查垂直间隔是否满足通行要求
    2. 根据梯度大小判断地形类型
    3. 对于中等梯度区域，检查周围是否有足够的平稳区域
    4. 计算相应的通行代价
    """
    trav_kernel = cp.ElementwiseKernel(
        in_params='raw U interval, raw U grad_mag_sq, raw U grad_mag_max',
        out_params='raw U trav_cost',
        preamble=utils_map(n_row, n_col),
        operation=string.Template(
            '''
            // Check minimum traversable interval
            if ( interval[i] < ${interval_min} )
            {
                trav_cost[i] = ${cost_barrier};
                return;
            }
            else
                trav_cost[i] += max(0.0, 20 * (${interval_free} - interval[i]));
                
            // Check if terrain is standable (low gradient)
            if ( grad_mag_sq[i] <= ${step_stand_sq} )
            {
                trav_cost[i] += 15 * grad_mag_sq[i] / ${step_stand_sq};
                return;
            }
            else 
            {
                // Check if terrain is crossable (medium gradient)
                if ( grad_mag_max[i] <= ${step_cross_sq} )
                {
                    // Count standable grids in neighborhood
                    int standable_grids = 0;
                    for ( int dy = -${half_kernel_size}; dy <= ${half_kernel_size}; dy++ ) 
                    {
                        for ( int dx = -${half_kernel_size}; dx <= ${half_kernel_size}; dx++ ) 
                        {
                            int idx = getIdxRelative(i, dx, dy);
                            if ( idx < 0 )
                                continue;
                            if ( grad_mag_sq[idx] < ${step_stand_sq} )
                                standable_grids += 1;
                        }
                    }
                    // Check if enough standable grids exist
                    if ( standable_grids < ${standable_th} )
                    {
                        trav_cost[i] = ${cost_barrier};
                        return;
                    }
                    else
                        trav_cost[i] += 20 * grad_mag_max[i] / ${step_cross_sq};
                }
                else
                {
                    // Too steep - mark as obstacle
                    trav_cost[i] = ${cost_barrier};
                    return;
                }
            } 
            '''
        ).substitute(
            half_kernel_size=half_kernel_size,
            interval_min=interval_min,
            interval_free=interval_free,
            step_cross_sq=step_cross ** 2,
            step_stand_sq=step_stand ** 2,
            standable_th=standable_th,
            cost_barrier=cost_barrier
        ),
        name='trav_kernel'
    )
                            
    return trav_kernel


def inflationKernel(n_row, n_col, half_kernel_size):
    """
    代价膨胀内核函数
    
    功能：
    - 对遍历代价进行膨胀操作，扩大障碍物的影响范围
    - 通过卷积操作将高代价区域的影响传播到周围
    - 提高路径规划的安全性，避免过于接近障碍物
    
    参数：
    - n_row: 网格行数
    - n_col: 网格列数  
    - half_kernel_size: 膨胀核的半径大小
    
    输入数据：
    - trav_cost: 原始遍历代价地图
    - score_table: 距离权重表，定义不同距离的影响权重
    
    输出数据：
    - inflated_cost: 膨胀后的代价地图
    
    膨胀逻辑：
    1. 遍历每个网格点的邻域
    2. 计算邻域内所有点的加权代价
    3. 取最大值作为当前点的膨胀代价
    4. 距离越近的点权重越大，影响越强
    """
    inflation_kernel = cp.ElementwiseKernel(
        in_params='raw U trav_cost, raw U score_table',
        out_params='raw U inflated_cost',
        preamble=utils_map(n_row, n_col),
        operation=string.Template(
            '''
            // Initialize counter for score table indexing
            int counter = 0;
            
            // Iterate through neighborhood kernel
            for ( int dy = -${half_kernel_size}; dy <= ${half_kernel_size}; dy++ ) 
            {
                for ( int dx = -${half_kernel_size}; dx <= ${half_kernel_size}; dx++ ) 
                {
                    // Get relative neighbor index
                    int idx = getIdxRelative(i, dx, dy);
                    if ( idx >= 0 )
                        // Apply weighted cost inflation (take maximum)
                        inflated_cost[i] = max(inflated_cost[i], trav_cost[idx] * score_table[counter]);
                    counter += 1;
                }
            }
            '''
        ).substitute(
            half_kernel_size=half_kernel_size
        ),
        name='inflation_kernel'
    )
                            
    return inflation_kernel