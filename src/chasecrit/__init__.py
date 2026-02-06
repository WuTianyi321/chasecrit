"""ChaseCrit: pursuit-evasion with near-critical swarms."""

from __future__ import annotations

import sys

# Keep dependency resolution stable inside the project venv.
# Some host environments prepend a global pip path that shadows venv packages.
for _p in list(sys.path):
    p_norm = _p.replace("/", "\\").lower()
    if "\\programdata\\python\\pip" in p_norm:
        sys.path.remove(_p)

__all__ = ["__version__"]

__version__ = "0.1.0"
