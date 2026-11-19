"""Private typed-source adapter; frozen v1 evidence and resource gates stay intact."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
TEST = RT / "test_salvage_current_resource_restore_v2.py"
FROZEN_ENTRY = RT / "prepare_salvage_current_resource_restore_v1.py"
FROZEN_TEST = RT / "test_salvage_current_resource_restore_v1.py"
FROZEN_ENTRY_SHA = "a5341f7c8f07077d001765b3795c01a6482fd2688d7c699d79b9ec94c8814fea"
FROZEN_TEST_SHA = "4357964c38f749aa8df2733ce22e92b551d71597be9c34102e4b9c62e6d2e303"
STAT_KEYS = ("st_dev", "st_ino", "st_uid", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")


def checked_source(reference):
    path = Path(reference["path"])
    if not path.is_absolute():
        raise RuntimeError("Immutable reference changed: " + str(path))
    before = path.stat()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read()
        finished = os.fstat(stream.fileno())
    after = path.stat()
    fingerprint = lambda st: tuple(getattr(st, key) for key in STAT_KEYS)
    if any(fingerprint(st) != fingerprint(before) for st in (opened, finished, after)):
        raise RuntimeError("File changed during full SHA: " + str(path))
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise RuntimeError("Immutable reference changed: " + str(path))
    return raw


def private_restore():
    raw = checked_source({"path": str(FROZEN_ENTRY), "sha256": FROZEN_ENTRY_SHA})
    checked_source({"path": str(FROZEN_TEST), "sha256": FROZEN_TEST_SHA})
    spec = importlib.util.spec_from_file_location("salvage_current_v2_private_frozen_v1", FROZEN_ENTRY)
    value = importlib.util.module_from_spec(spec)
    # Execute the already checked bytes, not a second unbound source read.
    exec(compile(raw, str(FROZEN_ENTRY), "exec"), value.__dict__)
    value.SELF, value.TEST = SELF, TEST
    value.PINNED = {**value.PINNED, FROZEN_ENTRY.name: FROZEN_ENTRY_SHA, FROZEN_TEST.name: FROZEN_TEST_SHA}
    value.checked_source = checked_source
    tree = ast.parse(raw, filename=str(FROZEN_ENTRY))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "safe_chain")
    expected = ast.dump(ast.parse("checked(cfg['resource_manager'])", mode="eval").body)
    matches = [node for node in ast.walk(function) if isinstance(node, ast.Call) and ast.dump(node) == expected]
    if len(matches) != 1:
        raise RuntimeError("Frozen safe_chain typed source call differs")
    # Only this explicitly typed reference changes; every JSON checked call stays.
    matches[0].func.id = "checked_source"
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(FROZEN_ENTRY), "exec"), value.__dict__)
    return value


restore = private_restore()


if __name__ == "__main__":
    restore.main()
