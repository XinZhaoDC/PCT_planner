"""Discover, load, validate, and select shared scene configurations."""

from dataclasses import replace
import importlib
import math
from pathlib import Path
import pkgutil
import re
from typing import List, Optional, Tuple

from . import scenes
from .model import RouteConfig, SceneConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCENE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SCENE_MODULE_PREFIX = "scene_"


def available_scenes() -> Tuple[str, ...]:
    names = []
    for module in pkgutil.iter_modules(scenes.__path__):
        if module.name.startswith(SCENE_MODULE_PREFIX):
            names.append(module.name[len(SCENE_MODULE_PREFIX):])
    return tuple(sorted(names))


def _resolve_repository_path(value: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _validate_coordinate(value: Tuple[float, ...], field_name: str) -> None:
    if len(value) != 2 or not all(math.isfinite(float(item)) for item in value):
        raise ValueError("%s must contain two finite coordinates" % field_name)


def _validate_scene(scene: SceneConfig, requested_name: str) -> None:
    if scene.name != requested_name:
        raise ValueError(
            "Scene module %r exports name %r" % (requested_name, scene.name)
        )
    if scene.default_route not in scene.routes:
        raise ValueError(
            "Scene %r default route %r is not defined"
            % (scene.name, scene.default_route)
        )
    if len(scene.waypoint_offset) != 3 or not all(
        math.isfinite(float(item)) for item in scene.waypoint_offset
    ):
        raise ValueError("Scene %r waypoint_offset must have three finite values" % scene.name)
    if scene.pcd_file.suffix.lower() != ".pcd":
        raise ValueError("Scene %r pcd_file must end with .pcd" % scene.name)
    if scene.tomogram_file.suffix.lower() != ".pickle":
        raise ValueError("Scene %r tomogram_file must end with .pickle" % scene.name)

    for route_name, route in scene.routes.items():
        if route_name == "all":
            raise ValueError("Scene %r cannot define the reserved route 'all'" % scene.name)
        if not route.waypoint_file:
            raise ValueError("Scene %r route %r has no waypoint file" % (scene.name, route_name))
        _validate_coordinate(route.start, "%s.%s.start" % (scene.name, route_name))
        _validate_coordinate(route.end, "%s.%s.end" % (scene.name, route_name))


def load_scene(name: str) -> SceneConfig:
    if not SCENE_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Scene names must use lowercase snake_case; received %r" % name
        )

    module_name = "%s.%s%s" % (scenes.__name__, SCENE_MODULE_PREFIX, name)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        available = ", ".join(available_scenes()) or "<none>"
        raise ValueError(
            "Unknown scene %r; available scenes: %s" % (name, available)
        ) from exc

    try:
        configured_scene = module.SCENE
    except AttributeError as exc:
        raise ValueError("%s must export SCENE" % module_name) from exc
    if not isinstance(configured_scene, SceneConfig):
        raise ValueError("%s.SCENE must be a SceneConfig" % module_name)

    resolved_scene = replace(
        configured_scene,
        pcd_file=_resolve_repository_path(configured_scene.pcd_file),
        tomogram_file=_resolve_repository_path(configured_scene.tomogram_file),
        waypoint_dir=_resolve_repository_path(configured_scene.waypoint_dir),
    )
    _validate_scene(resolved_scene, name)
    return resolved_scene


def select_routes(
    scene: SceneConfig, requested_route: Optional[str]
) -> List[Tuple[str, RouteConfig]]:
    if requested_route == "all":
        selected = [
            (name, route)
            for name, route in scene.routes.items()
            if route.include_in_all
        ]
        if not selected:
            raise ValueError("Scene %r has no routes enabled for 'all'" % scene.name)
        return selected

    route_name = requested_route or scene.default_route
    try:
        return [(route_name, scene.routes[route_name])]
    except KeyError as exc:
        available = ", ".join(scene.routes)
        raise ValueError(
            "Unknown route %r for scene %r; available routes: %s, all"
            % (route_name, scene.name, available)
        ) from exc
