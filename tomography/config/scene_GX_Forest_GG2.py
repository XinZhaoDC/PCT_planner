from .scene import ScenePCD, SceneMap, SceneTrav


class SceneGXForestGG2():
    pcd = ScenePCD()
    pcd.file_name = 'GXForestGG2.pcd'

    map = SceneMap()
    map.resolution = 0.50
    map.ground_h = 0.0    # 场景最低点
    map.slice_dh = 1.0

    trav = SceneTrav()
    trav.kernel_size = 0
    trav.interval_min = 1.7
    trav.interval_free = 1.7
    trav.slope_max = 10.0
    trav.step_max = 10.0
    trav.standable_ratio = 0.001
    trav.cost_barrier = 50.0
    trav.safe_margin = 0.0
    trav.inflation = 0.0
