from .scene import ScenePCD, SceneMap, SceneTrav


class SceneHeightOnly():
    """
    仅使用高度相关惩罚的场景配置
    
    特点：
    - 禁用坡度惩罚（设置极大的坡度阈值）
    - 禁用膨胀惩罚（设置膨胀半径为0）
    - 只保留高度间隔相关的代价计算
    """
    pcd = ScenePCD()
    pcd.file_name = 'XXX.pcd'  # 可以替换为你的点云文件

    map = SceneMap()
    map.resolution = 0.10
    map.ground_h = 0.0
    map.slice_dh = 0.5

    trav = SceneTrav()
    trav.kernel_size = 1
    trav.interval_min = 0.50        # 最小可通行间隔
    trav.interval_free = 0.65       # 自由通行间隔阈值
    
    # 设置极大的坡度阈值，实际禁用坡度惩罚
    trav.slope_max = 10.0           # 设置为极大值 (原来是 0.36)
    trav.step_max = 10.0            # 设置为极大值 (原来是 0.20)
    trav.standable_ratio = 0.01     # 设置为极小值，基本不考虑可站立区域
    
    trav.cost_barrier = 50.0        # 障碍物代价

    # 禁用膨胀惩罚
    trav.safe_margin = 0.0          # 缓冲区设为0
    trav.inflation = 0.0            # 膨胀区设为0
