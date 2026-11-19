"""Full-12000 N5 GPU6 authorization, with unchanged pidfd/lease/flock lifecycle."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("full12k_base_resource", Path(__file__).with_name("resource_manager.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def validate_roi(cfg):
    if (cfg.get("node") != "N5" or cfg.get("gpus") != [6]
            or cfg.get("release_authorized") is not True or cfg.get("phase") != "full12k_roi_v1"):
        raise RuntimeError("Only full-pool authorized N5 physical GPU6 extraction allowed")
    holders = cfg.get("holders", [])
    if len(holders) != 1 or holders[0].get("gpu") != 6 or cfg["env"].get("CUDA_VISIBLE_DEVICES") != "6":
        raise RuntimeError("One fresh exact N5 GPU6 holder required")
    if cfg.get("targetcount") != 12000 or cfg.get("input_count") != 12000:
        raise RuntimeError("Full ROI input must be exactly 12000 sources")
    if cfg.get("source_selection") != {"shard": 0, "nshard": 1, "limit": 0}:
        raise RuntimeError("ROI selection must consume the full authorized pool")
    if cfg["producer"]["path"] not in cfg["argv"] or "roi" not in cfg["argv"]:
        raise RuntimeError("Explicit frozen full-pool producer argv required")
    if not Path(cfg["cwd"]).is_absolute():
        raise RuntimeError("Absolute extraction cwd required")


def check_roi_model(cfg):
    ref = cfg["modelref"]
    if base.sha(ref["path"]) != ref["sha256"]:
        raise RuntimeError("Frozen ROI model reference changed")
    identity = json.loads(Path(ref["path"]).read_text())
    if identity["sha256"] != ref["model_identity_sha256"]:
        raise RuntimeError("ROI model identity mismatch")
    for value in identity["files"].values():
        if base.sha(value["path"]) != value["sha256"]:
            raise RuntimeError("ROI detector/pose/verified import bytes changed")


base.AUTHORIZED_PAIRS = {"N5": ((6,),)}
base.validate_pair = validate_roi
base.check_model = check_roi_model
if __name__ == "__main__":
    base.main()
