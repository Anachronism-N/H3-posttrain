"""Portable opt-in acceleration over native admitted facts and transactions.

Import with scripts_omnidance/exp/ms on sys.path. The caller owns prepared RGB
items, model/runtime configuration, resource ownership and production integration.
This module never loads a model, changes a token budget or retries a transaction.
CPU contracts do not demonstrate GPU throughput or human caption quality.
"""
from collections.abc import Mapping
import copy
from pathlib import Path

import ms_bar_annotation as bar

DIRECT_RECIPE = "caption_direct_admitted_facts_v1"
FLAGS = {**bar.FLAGS, "training_eligible": False}


def direct_caption(windows, facts, boundaries):
    """Join ALL already-admitted facts.text verbatim in each original window.

    Use native facts_from_observations output, including raw_text/part evidence.
    No support filtering, hand-detail removal, paraphrase or boundary inference
    happens here. Boundaries go to the FULL native validator but are never cited
    or turned into actions. Zero-fact windows have empty captions.

    Example after native independent observations have been admitted::

        facts = bar.facts_from_observations(windows, observations)
        body = direct_caption(windows, facts, boundaries=[])

    This is a new derivation, not the original model rewrite. Keep it in a fresh
    isolated output with an explicitly selected recipe; do not overwrite/resume
    an existing native-rewrite run or treat this candidate as training approval.
    """
    bar.need(isinstance(windows, (list, tuple)) and isinstance(facts, (list, tuple)),
             "ordered original windows and admitted facts required")
    bar.need(all(isinstance(w, dict) and bar.portable(w.get("window_id")) for w in windows),
             "invalid original window ID")
    window_ids = [w["window_id"] for w in windows]
    bar.need(len(set(window_ids)) == len(window_ids), "duplicate original window IDs")
    bar.need(all(isinstance(f, dict) and isinstance(f.get("fact_id"), str) and f["fact_id"]
                 and f.get("window_id") in window_ids and isinstance(f.get("text"), str)
                 and f["text"].strip() and isinstance(f.get("raw_text"), str) for f in facts),
             "admitted fact requires original window, ID, text and raw_text")
    fact_ids = [f["fact_id"] for f in facts]
    bar.need(len(set(fact_ids)) == len(fact_ids), "duplicate admitted fact IDs")
    result = {"windows": [], "issues": []}
    for window in windows:
        local = [f for f in facts if f["window_id"] == window["window_id"]]
        result["windows"].append({"window_id": window["window_id"],
            "caption": " ".join(f["text"] for f in local),
            "used_fact_ids": [f["fact_id"] for f in local], "used_boundary_ids": []})
    return bar.check_supported_rewrite(result, windows, facts, boundaries)


def direct_final(windows, facts, boundaries):
    """Wrap direct_caption as an unreviewed, explicitly new, no-model candidate."""
    metadata = {"derivation": DIRECT_RECIPE, "model_call_performed": False,
                "model_rewrite_performed": False, "semantic_validation_completed": False, **FLAGS}
    try:
        body = direct_caption(windows, facts, boundaries)
    except ValueError as error:
        return {"status": "failed", "error": f"{type(error).__name__}: {error}",
                "format_valid": False, **metadata}
    return {"status": "candidate" if facts else "abstain", "json": body,
            "format_valid": True, **metadata}


def cross_source_transactions(source_items, runtime, args, batch_size=8):
    """Dispatch prepared independent visual windows across sources in ONE call.

    source_items is an insertion-ordered mapping: source ID -> ordered native
    window items (root, request, validator, binding). Bindings must contain the
    matching source_binding.id, original window and frames_sha256. Each validator
    is the caller's native support-filter closure over that window's evidence.
    Windows with fewer than two frames stay caller-owned evidence abstentions;
    never fabricate a request for them. No boundaries or rewrites are accepted.

    Example, with a caller-owned already configured/shared runtime::

        source_items = {source_id: prepared_window_items,
                        another_source_id: another_prepared_window_items}
        grouped = cross_source_transactions(source_items, runtime, args, 8)
        result = grouped["by_root"][str(prepared_window_items[0]["root"])]
        source_results = grouped["by_source"][source_id]

    All validation precedes disk writes/runtime use. Flattening preserves mapping
    and per-source order. Only a shallow args copy's batch_size changes; the
    caller must configure max_num_seqs >= batch_size. Native messages (including
    JPEG bytes), schemas, sampling, locks, orphan-raw handling and receipts pass
    through unchanged. by_root is the native root-result mapping; by_source maps
    each source ID to its ordered subset of those SAME results. This interface
    does not copy the consumer loop, resource manager or transaction cache.
    """
    bar.need(isinstance(source_items, Mapping), "source_items must be a source mapping")
    bar.need(type(batch_size) is int and batch_size > 0, "batch_size must be a positive integer")
    bar.need(getattr(args, "stage", None) in ("prepare", "run"), "only prepare/run visual stages supported")
    bar.need(type(getattr(args, "max_num_seqs", None)) is int and args.max_num_seqs >= batch_size,
             "caller max_num_seqs must be at least batch_size")
    items, roots, source_windows, source_roots = [], set(), set(), {}
    for sid, windows in source_items.items():
        bar.need(bar.portable(sid) and isinstance(windows, (list, tuple)), "ordered source window items required")
        source_roots[sid] = []
        for item in windows:
            bar.need(isinstance(item, dict) and {"root", "request", "validator", "binding"} <= set(item),
                     "native window transaction item required")
            binding, request = item["binding"], item["request"]
            bar.need(isinstance(binding, dict) and isinstance(binding.get("source_binding"), dict)
                     and binding["source_binding"].get("id") == sid, "source binding ID mismatch")
            window = binding.get("window")
            bar.need(isinstance(window, dict) and bar.portable(window.get("window_id"))
                     and bar.valid_sha(binding.get("frames_sha256")), "original window/frame binding required")
            bar.core.window_bounds(window)
            bar.need(not {"boundary", "reference_repair_sha256"} & set(binding), "dependent stage binding not allowed")
            root = Path(item["root"])
            identity, source_window = root.resolve(), (sid, window["window_id"])
            bar.need(identity not in roots and source_window not in source_windows, "duplicate root or source/window")
            bar.need(callable(item["validator"]), "native window validator required")
            bar.need(isinstance(request, dict) and set(request) ==
                     {"model", "messages", "sampling", "json_schema", "prompt_sha256"},
                     "independent native visual request required")
            messages = request["messages"]
            bar.need(isinstance(messages, list) and len(messages) == 2
                     and messages[0] == {"role": "system", "content": bar.SYSTEM}
                     and isinstance(messages[1], dict) and messages[1].get("role") == "user",
                     "only native visual system/user messages allowed")
            content = messages[1].get("content")
            bar.need(isinstance(content, list) and content and isinstance(content[0], dict)
                     and content[0].get("type") == "text" and isinstance(content[0].get("text"), str),
                     "native visual frame header required")
            header = bar.core.strict_json(content[0]["text"])
            indices = header.get("allowed_source_frame_indices") if isinstance(header, dict) else None
            bar.need(isinstance(indices, list) and all(type(i) is int and i >= 0 for i in indices),
                     "actual source frame indices required")
            bar.check_indices(indices, set(indices), 2)
            bar.need(request["json_schema"] == bar.observation_schema(indices), "native visual schema differs")
            roots.add(identity)
            source_windows.add(source_window)
            source_roots[sid].append(str(root))
            items.append({**item, "root": root})
    copied_args = copy.copy(args)
    copied_args.batch_size = batch_size
    results = bar.transactions(items, runtime, copied_args)
    return {"by_root": results, "by_source": {
        sid: {root: results[root] for root in paths} for sid, paths in source_roots.items()}}
