"""Whole-response candidate selection without model calls or hidden resampling.

The two captions in one expert response are one indivisible choice.  Hashes
bind reproducible evidence; they do not authenticate the expert's semantics.
"""
import copy
import math

from .candidates import CandidateBank, _validate_bank, audio_identity
from .io import finite_json, object_sha, read_json_snapshot
from .schema import integer, keys, text, validate_song


class NoFeasibleSelectionError(ValueError):
    """Well-formed fixed scores admit no fully known complete candidate path."""


def _seal(value):
    value = copy.deepcopy(value)
    value["record_sha256"] = object_sha(value)
    return value


def _verify(value):
    if not isinstance(value, dict) or value.get("record_sha256") != object_sha(
            {k: v for k, v in value.items() if k != "record_sha256"}):
        raise ValueError("Selection/options content hash differs")


def _song_evidence(bank, song):
    """Validate a complete source song, not a convenient valid prefix."""
    if not isinstance(bank, CandidateBank):
        raise ValueError("A loaded CandidateBank is required")
    validate_song(song)
    bank.assert_unchanged()
    data, bytes_sha = read_json_snapshot(bank.path)
    if bytes_sha != bank.bytes_sha or object_sha(data) != object_sha(bank.data):
        raise ValueError("Candidate bank bytes or in-memory content hash differs")
    # Revalidate the exact file snapshot, not mutable cached entry references.
    entries, states, _ = _validate_bank(data)
    originals = data["source"]["contract"]["dataset_snapshot"]["songs"]
    original = next((s for s in originals if s["song_id"] == song["song_id"]), None)
    if original is None or object_sha(original) != object_sha(song):
        raise ValueError("Current song snapshot differs from candidate source")
    state = next((s for s in states if s["song_id"] == song["song_id"]), None)
    if state is None or state["status"] != "complete":
        from .baseline import ModelOutputError
        raise ModelOutputError("Source song has a recorded local model failure; its prefix cannot be selected")
    binding = dict(bank_file_sha256=bank.bytes_sha, bank_record_sha256=data["record_sha256"],
        bank_binding_sha256=object_sha(data["binding"]),
        source_contract_sha256=object_sha(data["source"]["contract"]),
        dataset_sha256=data["binding"]["dataset_sha256"], song_sha256=object_sha(song),
        audio=audio_identity(song["audio"]),
        local_parser_sha256=data["binding"]["local_parser_sha256"],
        provider_identity_sha256=object_sha(data["binding"]["provider_identity"]))
    selected = [entry for entry in entries.values() if entry["song_id"] == song["song_id"]]
    return binding, selected, data["synthetic"], data["binding"]["provider_identity"]


def _public_units(units, duration):
    if units is None:
        return []
    if isinstance(units, dict):
        units = units.get("units")
    if not isinstance(units, list):
        raise ValueError("units must be a list or a sidecar containing units")
    result, seen = [], set()
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("Planning unit must be an object")
        text(unit.get("unit_id"), "unit_id")
        integer(unit.get("start"), "unit.start")
        integer(unit.get("end"), "unit.end", 1)
        if unit["unit_id"] in seen or not unit["start"] < unit["end"] <= duration:
            raise ValueError("Duplicate/out-of-range planning unit")
        seen.add(unit["unit_id"])
        row = {k: copy.deepcopy(unit[k]) for k in ("unit_id", "start", "end", "kind", "section_id") if k in unit}
        for name in ("kind", "section_id"):
            if name in row and row[name] is not None:
                text(row[name], "unit." + name)
        finite_json(row)
        result.append(row)
    return result


def candidate_options(bank, song, units=None):
    """Public whole-pair options; never expose group/split/path/manifest data.

    Optional units add overlap references only, not a new candidate decision.
    Nominal intervals normalize the legacy parser's explicitly retained timing
    tolerance to the sample clock; none are observed motion boundaries.
    """
    from .baseline import _local_actions, _recorded_result
    binding, entries, synthetic, provider_identity = _song_evidence(bank, song)
    public_units = _public_units(units, song["audio"]["num_samples"])
    windows = []
    for entry in entries:
        window = entry["window"]
        if not windows or windows[-1]["window_id"] != "w%04d" % window["index"]:
            windows.append(dict(window_id="w%04d" % window["index"], window=copy.deepcopy(window),
                candidates=[], unit_refs=[u["unit_id"] for u in public_units
                    if window["observation_start"] < u["end"] and window["observation_end"] > u["start"]]))
        result = _recorded_result(entry["raw"], provider_identity, entry["started"]["request"])
        parsed = _local_actions(result["text"])
        start, end = window["observation_start"], window["observation_end"]
        split = min(end - start, int(math.floor(2. * song["audio"]["sample_rate"] + .5)))
        bounds = ((start, start + split), (start + split, end))
        fragments = []
        for i, caption in enumerate(parsed["windows"]):
            left = max(bounds[i][0], window["ownership_start"])
            right = min(bounds[i][1], window["ownership_end"])
            inside = left == bounds[i][0] and right == bounds[i][1]
            fragments.append(dict(part_index=i, text=caption["text"],
                declared_interval_seconds=[caption["start"], caption["end"]], nominal_interval=list(bounds[i]),
                ownership_intersection=[left, right] if left < right else None,
                fully_inside_ownership=inside, partial_source=left < right and not inside))
        windows[-1]["candidates"].append(dict(candidate_index=entry["candidate_index"], call_id=entry["call_id"],
            raw_text=result["text"], raw_text_sha256=object_sha(result["text"]),
            raw_record_sha256=entry["raw"]["record_sha256"], request_sha256=entry["started"]["request_sha256"],
            fragments=fragments))
    result = _seal(dict(schema="choreobench.candidate_options.v1", song_id=song["song_id"],
        sample_rate=song["audio"]["sample_rate"], num_samples=song["audio"]["num_samples"],
        synthetic=synthetic, binding=binding, units=public_units, windows=windows))
    bank.assert_unchanged(full=True)
    return result


def _selection_document(options, choices, selection_provenance=None):
    if selection_provenance is not None:
        if not isinstance(selection_provenance, dict):
            raise ValueError("selection_provenance must be an object or null")
        text(selection_provenance.get("source"), "selection_provenance.source")
        if type(selection_provenance.get("synthetic")) is not bool:
            raise ValueError("selection_provenance.synthetic must be explicit boolean")
        finite_json(selection_provenance)
    if not isinstance(choices, list):
        raise ValueError("choices must be a complete list of whole-window decisions")
    expected = {row["window_id"]: row for row in options["windows"]}
    by_window = {}
    for choice in choices:
        keys(choice, ("window_id", "candidate_index"), where="choice")
        wid, index = choice["window_id"], choice["candidate_index"]
        if not isinstance(wid, str) or wid not in expected or wid in by_window:
            raise ValueError("Unknown or duplicate selection window_id")
        integer(index, "candidate_index")
        candidate = next((c for c in expected[wid]["candidates"] if c["candidate_index"] == index), None)
        if candidate is None:
            raise ValueError("Unknown candidate_index for window " + wid)
        by_window[wid] = candidate
    if set(by_window) != set(expected):
        raise ValueError("Selection must choose exactly one complete candidate for every window")
    ordered, sources = [], []
    for row in options["windows"]:
        candidate = by_window[row["window_id"]]
        ordered.append(dict(window_id=row["window_id"], candidate_index=candidate["candidate_index"]))
        sources.append(dict(window_id=row["window_id"], call_id=candidate["call_id"],
            request_sha256=candidate["request_sha256"], raw_record_sha256=candidate["raw_record_sha256"],
            raw_text_sha256=candidate["raw_text_sha256"]))
    value = dict(schema="choreobench.selection.v1", binding=copy.deepcopy(options["binding"]),
        song_id=options["song_id"], synthetic=bool(options["synthetic"] or
            (selection_provenance is not None and selection_provenance["synthetic"])), choices=ordered, sources=sources)
    if selection_provenance is not None:
        value["selection_provenance"] = copy.deepcopy(selection_provenance)
    return _seal(value)


def make_selection(bank, song, choices, *, selection_provenance=None):
    """Seal a complete selection bound to bank bytes, raw records and song."""
    return _selection_document(candidate_options(bank, song), choices, selection_provenance)


def _validated_selection(bank, song, selection):
    _verify(selection)
    options = candidate_options(bank, song)
    expected = _selection_document(options, selection.get("choices"), selection.get("selection_provenance"))
    if object_sha(selection) != object_sha(expected):
        raise ValueError("Selection source/song/audio/parser/bank binding differs")
    return options


def validate_selection(document, bank, song):
    """Validate complete current source binding; return the original document."""
    _validated_selection(bank, song, document)
    return document


def select_with_scores(options, unary, transitions, *, score_provenance):
    """Exact first-order DP for fixed scores; null means unavailable/forbidden.

    unary[i][k]; transitions[i][j][k] connects layer i candidate j to i+1 k.
    Equal totals choose the lexicographically smallest candidate-index path.
    Unknown mandatory scores are never filled with zero. No quality evaluator
    or scoring provider is created here. Transition work is O(N*K**2).
    """
    _verify(options)
    if options.get("schema") != "choreobench.candidate_options.v1":
        raise ValueError("Expected candidate_options output")
    if not isinstance(score_provenance, dict):
        raise ValueError("Explicit score_provenance is required")
    text(score_provenance.get("source"), "score_provenance.source")
    if type(score_provenance.get("synthetic")) is not bool:
        raise ValueError("score_provenance.synthetic must be explicit boolean")
    finite_json(score_provenance)
    windows = options.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Nonempty candidate layers required")
    indices = [[c["candidate_index"] for c in row["candidates"]] for row in windows]
    if any(not row or len(set(row)) != len(row) or any(type(k) is not int or k < 0 for k in row) for row in indices):
        raise ValueError("Candidate layers must contain unique integer indices")
    if not isinstance(unary, list) or len(unary) != len(windows):
        raise ValueError("unary must have one row per window")
    if not isinstance(transitions, list) or len(transitions) != len(windows) - 1:
        raise ValueError("transitions must have one matrix per adjacent window pair")
    def number(value):
        if value is None:
            return None
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Scores must be finite numbers or explicit null")
        return value
    for i, row in enumerate(unary):
        if not isinstance(row, list) or len(row) != len(indices[i]):
            raise ValueError("unary candidate dimensions differ")
        for value in row:
            number(value)
    for i, matrix in enumerate(transitions):
        if (not isinstance(matrix, list) or len(matrix) != len(indices[i])
                or any(not isinstance(row, list) or len(row) != len(indices[i + 1]) for row in matrix)):
            raise ValueError("transition candidate dimensions differ")
        for row in matrix:
            for value in row:
                number(value)
    values = list(unary[0])
    ranks = {k: rank for rank, k in enumerate(sorted((k for k, v in enumerate(values) if v is not None),
                                                    key=lambda k: indices[0][k]))}
    parents = []
    for i in range(1, len(windows)):
        next_values, previous = [None] * len(indices[i]), [None] * len(indices[i])
        for k, own in enumerate(unary[i]):
            if own is None:
                continue
            for j, prior in enumerate(values):
                edge = transitions[i - 1][j][k]
                if prior is None or edge is None:
                    continue
                total = prior + edge + own
                try:
                    finite_total = math.isfinite(total)
                except OverflowError:
                    finite_total = False
                if not finite_total:
                    raise ValueError("Accumulated score is nonfinite")
                if (next_values[k] is None or total > next_values[k]
                        or (total == next_values[k] and ranks[j] < ranks[previous[k]])):
                    next_values[k], previous[k] = total, j
        ordered = sorted((k for k, v in enumerate(next_values) if v is not None),
                         key=lambda k: (ranks[previous[k]], indices[i][k]))
        ranks = {k: rank for rank, k in enumerate(ordered)}
        parents.append(previous)
        values = next_values
    available = [k for k, value in enumerate(values) if value is not None]
    if not available:
        raise NoFeasibleSelectionError("No complete path with known mandatory scores; explicit abstention required")
    best = min(available, key=lambda k: (-values[k], ranks[k]))
    total = values[best]
    path = [best]
    for previous in reversed(parents):
        path.append(previous[path[-1]])
    path.reverse()
    choices = [dict(window_id=row["window_id"], candidate_index=indices[i][path[i]])
               for i, row in enumerate(windows)]
    return _seal(dict(schema="choreobench.scored_selection.v1", choices=choices, total_score=total,
        options_sha256=object_sha(options), scores_sha256=object_sha(dict(unary=unary, transitions=transitions)),
        score_provenance=copy.deepcopy(score_provenance),
        synthetic=bool(options["synthetic"] or score_provenance["synthetic"]),
        tie_policy="lexicographically_smallest_candidate_index_path", objective="fixed_unary_plus_adjacent_pair",
        music_or_physical_quality_verified=False))
