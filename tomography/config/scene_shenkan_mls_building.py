from .scene import ScenePCD, SceneMap, SceneTrav

class SceneShenkanMLS():
    pcd = ScenePCD()
    pcd.file_name = 'shenkan_MLS_building.pcd'

    map = SceneMap()
    map.resolution = 0.3
    map.ground_h = 1.0
    map.slice_dh = 1.0

    trav = SceneTrav()
    trav.kernel_size = 1
    trav.interval_min = 1.6
    trav.interval_free = 1.7
    trav.slope_max = 1.0
    trav.step_max = 1.0
    trav.standable_ratio = 0.0
    trav.cost_barrier = 50.0
    trav.safe_margin = 0.0
    trav.inflation = 0.05
