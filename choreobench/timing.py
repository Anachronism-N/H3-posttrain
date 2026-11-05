"""Explicit proposed-event clocks; never inferred action or beat boundaries."""
from fractions import Fraction


class TimingInfeasibleError(ValueError):
    """A real planning unit cannot accommodate the declared minimum duration."""


def event_clock(unit, sample_rate, config):
    """Map optional local boundary IDs to the unchanged canonical sample clock.

    Grid steps round to nearest sample, ties upward. Minimum duration rounds
    upward. The real unit endpoint is always included; no padding or retiming.
    """
    minimum = Fraction(str(config.get("min_event_seconds", 0))) * sample_rate
    minimum = max(1, -(-minimum.numerator // minimum.denominator))
    if unit["end"] - unit["start"] < minimum:
        raise TimingInfeasibleError("Planning unit is shorter than min_event_seconds; no action fabricated")
    mode = config.get("event_time_format", "global_samples")
    result = dict(schema="choreobench.event_clock.v1", format=mode,
                  minimum_duration_samples=minimum, sample_rate=sample_rate,
                  boundaries_are_action_boundaries=False, beat_alignment_verified=False)
    if mode == "boundary_ids":
        product = Fraction(str(config.get("event_grid_seconds", .25))) * sample_rate
        step = (2 * product.numerator + product.denominator) // (2 * product.denominator)
        if step < 1:
            raise ValueError("event_grid_seconds must represent at least one sample")
        count = (unit["end"] - unit["start"] + step - 1) // step + 1
        if count > 4097:
            raise ValueError("Event grid exceeds 4097 boundaries; choose a coarser grid")
        result.update(boundary_samples=list(range(unit["start"], unit["end"], step)) + [unit["end"]],
                      step_samples=step, terminal_interval_may_be_shorter=True)
    elif mode != "global_samples":
        raise ValueError("Unknown event_time_format")
    return result


def event_interval(event, clock):
    """Compile an explicit boundary pair, then enforce the declared duration."""
    if clock["format"] == "boundary_ids":
        start, end = event["start_boundary"], event["end_boundary"]
        samples = clock["boundary_samples"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end < len(samples)):
            raise ValueError("Invalid or reversed event boundary IDs")
        start, end = samples[start], samples[end]
    else:
        start, end = event["start"], event["end"]
        if type(start) is not int or type(end) is not int:
            raise ValueError("Event times must be integer samples")
    if end - start < clock["minimum_duration_samples"]:
        raise ValueError("Proposed event is shorter than min_event_seconds; no automatic stretching")
    return start, end


def literal_source_copies(plan, options):
    """Exact strings only; equality and inequality say nothing about semantics."""
    inputs = {}
    for window in options["windows"]:
        for candidate in window["candidates"]:
            inputs[candidate["call_id"]] = {candidate["raw_text"]} | {
                fragment["text"] for fragment in candidate["fragments"]}
    matches = [e["event_id"] for e in plan["events"]
               if any(e["action"] in inputs.get(ref, set()) for ref in e["evidence"])]
    count = len(plan["events"])
    return dict(event_count=count, exact_match_count=len(matches), exact_match_fraction=len(matches) / count if count else None,
                matching_event_ids=matches,
                comparison="Exact Unicode action string against cited raw responses or parsed fragments",
                semantic_novelty_verified=False)
