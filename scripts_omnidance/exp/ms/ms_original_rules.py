"""Import unchanged upstream MotionScript from its flat source layout."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types


def load_original_rules(source):
    source = Path(source).resolve()
    if not (source / "captioning.py").is_file():
        raise ValueError("Original source must contain captioning.py")
    existing = sys.modules.get("text2pose.posescript.captioning")
    if existing is not None:
        if Path(existing.__file__).resolve() != source / "captioning.py":
            raise ValueError("A different MotionScript implementation is already imported")
        return existing
    if "text2pose" in sys.modules or "text2pose.posescript" in sys.modules:
        raise ValueError("Use a separate process from other text2pose installations")
    package = types.ModuleType("text2pose")
    package.__path__ = []
    posescript = types.ModuleType("text2pose.posescript")
    posescript.__path__ = [str(source)]
    package.posescript = posescript
    sys.modules["text2pose"] = package
    sys.modules["text2pose.posescript"] = posescript
    sys.path.insert(0, str(source))
    module = importlib.import_module("text2pose.posescript.captioning")
    if Path(module.__file__).resolve() != source / "captioning.py":
        raise ValueError("Imported original rules have the wrong provenance")
    return module
