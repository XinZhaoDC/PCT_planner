from pathlib import Path
import unittest

from pct_config import available_scenes, load_scene, select_routes


EXPECTED_SCENES = {
    "spiral",
    "building",
    "plaza",
    "whu_tls_forest",
    "shenkan_mls",
    "gx_forest_gg2",
    "gx_forest_gg3",
}

EXPECTED_MAP = {
    "spiral": (0.20, 0.0, 0.5),
    "building": (0.05, 0.0, 0.5),
    "plaza": (0.10, 0.0, 0.5),
    "whu_tls_forest": (0.10, 5.0, 0.5),
    "shenkan_mls": (0.3, 1.0, 1.0),
    "gx_forest_gg2": (0.50, 0.0, 1.0),
    "gx_forest_gg3": (0.50, 100.0, 1.0),
}

EXPECTED_TRAV = {
    "spiral": (7, 0.50, 0.65, 0.40, 0.30, 0.40, 50.0, 1.2, 0.2),
    "building": (7, 0.50, 0.65, 0.40, 0.17, 0.20, 50.0, 0.4, 0.2),
    "plaza": (7, 0.50, 0.65, 0.36, 0.17, 0.2, 50.0, 0.4, 0.2),
    "whu_tls_forest": (3, 1.6, 1.7, 1.0, 1.0, 0.0, 50.0, 0.0, 0.05),
    "shenkan_mls": (1, 1.6, 1.7, 1.0, 1.0, 0.0, 50.0, 0.0, 0.05),
    "gx_forest_gg2": (0, 1.7, 1.7, 10.0, 10.0, 0.001, 50.0, 0.0, 0.0),
    "gx_forest_gg3": (0, 1.7, 1.7, 10.0, 10.0, 0.001, 50.0, 0.0, 0.0),
}


class SceneConfigTest(unittest.TestCase):
    def test_discovery_uses_only_lowercase_runnable_scenes(self):
        self.assertEqual(set(available_scenes()), EXPECTED_SCENES)
        self.assertNotIn("height_only", available_scenes())

    def test_all_original_map_and_traversability_values_are_preserved(self):
        for name in EXPECTED_SCENES:
            scene = load_scene(name)
            self.assertEqual(
                (scene.map.resolution, scene.map.ground_h, scene.map.slice_dh),
                EXPECTED_MAP[name],
            )
            self.assertEqual(
                (
                    scene.trav.kernel_size,
                    scene.trav.interval_min,
                    scene.trav.interval_free,
                    scene.trav.slope_max,
                    scene.trav.step_max,
                    scene.trav.standable_ratio,
                    scene.trav.cost_barrier,
                    scene.trav.safe_margin,
                    scene.trav.inflation,
                ),
                EXPECTED_TRAV[name],
            )

    def test_paths_and_three_axis_offsets_are_scene_owned(self):
        for name in EXPECTED_SCENES:
            scene = load_scene(name)
            self.assertTrue(scene.pcd_file.is_absolute())
            self.assertTrue(scene.tomogram_file.is_absolute())
            self.assertTrue(scene.waypoint_dir.is_absolute())
            self.assertEqual(scene.pcd_file.suffix, ".pcd")
            self.assertEqual(scene.tomogram_file.suffix, ".pickle")
            self.assertEqual(scene.waypoint_offset, (0.0, 0.0, 1.6))

    def test_whu_default_and_all_routes_preserve_legacy_behavior(self):
        scene = load_scene("whu_tls_forest")
        default = select_routes(scene, None)
        self.assertEqual(default[0][0], "default")
        self.assertEqual(default[0][1].start, (13.40, 19.27))
        self.assertEqual(default[0][1].waypoint_file, "WHU_TLS_Forest_waypoint")

        all_routes = select_routes(scene, "all")
        self.assertEqual([name for name, _ in all_routes], ["1", "2", "3", "4", "5"])
        self.assertEqual(
            [route.waypoint_file for _, route in all_routes],
            ["waypoint_1", "waypoint_2", "waypoint_3", "waypoint_4", "waypoint_5"],
        )

    def test_historical_gx_routes_are_executable(self):
        for name in ("gx_forest_gg2", "gx_forest_gg3"):
            routes = select_routes(load_scene(name), "all")
            self.assertEqual([route_name for route_name, _ in routes], ["1", "2", "3", "4"])

    def test_uppercase_and_unknown_names_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "lowercase snake_case"):
            load_scene("WHUTLSForest")
        with self.assertRaisesRegex(ValueError, "available scenes"):
            load_scene("not_a_scene")

    def test_historical_coordinate_comments_are_retained(self):
        scene_dir = Path(__file__).resolve().parents[1] / "pct_config" / "scenes"
        expected_fragments = {
            "scene_whu_tls_forest.py": [
                "备选起终点坐标 (用于测试不同路径)",
                "[9.19,26.52] -> [48.18,54.52]",
            ],
            "scene_shenkan_mls.py": [
                "[507.9080 115.0592 1.4566] -> [54.6010 20.4289 1.4659]",
                "[54 20] -> [507 115] (当前使用)",
            ],
            "scene_gx_forest_gg2.py": [
                "[-57.879,51.645] -> [68.674,151.844]",
                "[-14.333,1.933] -> [29.644,188.636]",
            ],
            "scene_gx_forest_gg3.py": [
                "[-88.532,-67.000] -> [85.343,62.750]",
                "[71.532,-80.500] -> [-36.532,76.500]",
            ],
        }
        for file_name, fragments in expected_fragments.items():
            content = (scene_dir / file_name).read_text(encoding="utf-8")
            for fragment in fragments:
                self.assertIn(fragment, content)


if __name__ == "__main__":
    unittest.main()
