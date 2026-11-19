"""Caller-owned source locks/ready loop; native observation receipts before source resume.

No source completion, manifest publication, resource acquisition, or recipe approval
is performed here. The caller must keep the entries/configuration fixed and check
source input boundaries again before committing its completion notice.
"""
import copy
from pathlib import Path
import subprocess

from caption_direct_facts_v1 import native


class StagedSources:
    def __init__(self, entries, configuration):
        self.entries = tuple(entries)
        self.inputs = copy.deepcopy(self.entries)
        self.configuration = configuration
        self.errors = [None] * len(self.entries)
        self.receipts = {}


def check_inputs(staged, args):
    if vars(args) != staged.configuration or staged.entries != staged.inputs:
        raise ValueError("Staged source inputs or configuration changed")


def stage_sources(entries, args, runtime, ref_cache):
    """Flatten caller-ready {row, raw_line, root, binding} entries into one native batch.

    Use inside the caller's isolated recipe context and source locks. retry_failed
    must be false: otherwise native resume could reserve another attempt after a
    newly failed stage receipt. Preparation failures stay per-source exceptions,
    not prematurely published source results.
    """
    if args.stage != "run" or args.retry_failed:
        raise ValueError("Cross-source staging requires stage=run and retry_failed=False")
    staged = StagedSources(entries, copy.deepcopy(vars(args)))
    roots = [Path(e["root"]).resolve() for e in staged.entries]
    if len(roots) != len(set(roots)):
        raise ValueError("Duplicate caller source roots")
    bar, _ = native()
    items = []
    for index, entry in enumerate(staged.inputs):
        row, raw_line, binding = entry["row"], entry["raw_line"], entry["binding"]
        root = Path(entry["root"])
        local = []
        try:
            if (root / "prepared.json").exists():
                prepared = bar.verify_prepared(root, row, raw_line, binding)
            else:
                prepared = bar.prepare_source(row, raw_line, root, args, binding, ref_cache)
            forbidden = {p: set(v) for p, v in prepared["forbidden"].items()}
            for manifest in prepared["windows"]:
                wroot = root / "windows" / manifest["window"]["window_id"]
                bar.core.save_json(wroot / "manifest.json", {**manifest, "binding": binding,
                    "union_manifest_sha256": prepared["union_manifest_sha256"],
                    "raw2d_mask": prepared["raw2d_mask"], **bar.FLAGS})
                if len(manifest["frames"]) < 2:
                    continue
                local.append({"root": bar.attempt_root(wroot, args.retry_failed),
                    "request": bar.visual_request(args, root / "rgb", manifest, forbidden),
                    "validator": lambda value, m=manifest, f=forbidden:
                        bar.filter_supported_observation(value, m, f),
                    "binding": {**binding, "window": manifest["window"],
                                "frames_sha256": bar.batch.digest(manifest["frames"])}})
        except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
            staged.errors[index] = error
            continue
        items.extend(local)
    if len({str(item["root"]) for item in items}) != len(items):
        raise ValueError("Duplicate native window attempt roots")
    check_inputs(staged, args)
    staged.receipts = bar.transactions(items, runtime, args)
    return staged


def resume_source(staged, index, args, runtime, ref_cache):
    """Run original process_source only after the flattened transaction barrier.

    Native prepare verification and receipt support-filter replay are intentional.
    Boundary/rewrite behavior belongs to the caller's current recipe, not staging.
    """
    check_inputs(staged, args)
    if staged.errors[index] is not None:
        raise staged.errors[index]
    entry = staged.inputs[index]
    bar, _ = native()
    return bar.process_source(entry["row"], entry["raw_line"], Path(entry["root"]),
                              args, entry["binding"], runtime, ref_cache)
