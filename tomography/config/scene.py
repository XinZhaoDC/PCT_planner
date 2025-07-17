class ScenePCD():
    file_name = None


class SceneMap():
    resolution = 0.10 # 格网分辨率
    ground_h = 0.0 # 场景最低点
    slice_dh = 0.5 # 切片高度间隔  d_s


class SceneTrav():
    kernel_size = 7 # 机器人占地格网大小
    interval_min = 0.50 # 最小机器人高度 d_min
    interval_free = 0.65 # 正常状态下的机器人高度 d_ref
    slope_max = 0.36 # 梯度阈值2 theta_s（不是直接使用的这个值）
    step_max = 0.20 # 梯度阈值1 theta_b（不是直接使用的这个值）
    standable_ratio = 0.20 # 可行走区域的占比 theta_p

    cost_barrier = 50.0 # 障碍物代价 C^B

    safe_margin = 0.4 # 缓冲区大小 d_sm
    inflation = 0.2 # 膨胀区大小 d_inf