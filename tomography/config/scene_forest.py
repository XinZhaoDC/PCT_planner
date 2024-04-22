from .scene import ScenePCD, SceneMap, SceneTrav


class SceneForest():
    pcd = ScenePCD()
    pcd.file_name = 'path_planning.pcd'

    map = SceneMap()
    map.resolution = 0.10
    map.ground_h = 5.0    # 大于森林地面的最高处
    map.slice_dh = 0.5

    trav = SceneTrav()
    trav.kernel_size = 3
    trav.interval_min = 1.6
    trav.interval_free = 1.7
    trav.slope_max = 1.0
    trav.step_max = 1.0
    trav.standable_ratio = 0.0
    trav.cost_barrier = 50.0
    trav.safe_margin = 0.0
    trav.inflation = 0.05
