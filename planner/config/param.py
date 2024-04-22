class ConfigPlanner():
    use_quintic = True
    max_heading_rate = 10


class ConfigWrapper():
    tomo_dir = '/rsc/tomogram/'
    waypoint_dir = '/rsc/waypoint/'


class Config():
    planner = ConfigPlanner()
    wrapper = ConfigWrapper()