"""Small YAML loader with recursive ``include`` support."""

from copy import deepcopy
from pathlib import Path

import yaml


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path):
    """Load a config and merge included files in declaration order.

    Include paths are relative to the file that declares them.  Values in the
    including file override included values.
    """
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        current = yaml.safe_load(stream) or {}
    includes = current.pop("include", [])
    if isinstance(includes, (str, Path)):
        includes = [includes]
    merged = {}
    for include in includes:
        merged = _merge(merged, load_config(path.parent / include))
    return _merge(merged, current)


def project_root():
    return Path(__file__).resolve().parents[1]


def resolve_project_path(value):
    """Resolve a config path; relative values are project-root relative."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()
