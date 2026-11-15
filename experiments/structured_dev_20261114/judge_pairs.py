#!/usr/bin/env python3
"""Independent-model blind pairwise review of the S/R pairs.

The judge model is a different, smaller checkpoint than the generator, sees only
the blinded plans (A/B order already swapped by the report step) and the public
acoustic summary, and must cite event ids. This is model-assisted evidence, not
human review, and it is never fed back into generation.
"""
import argparse
import json
import urllib.request
from pathlib import Path

CRITERIA = ("musical_fit", "motif_development", "action_executability", "boundary_connection")
SYSTEM = (
    "You compare two anonymous timed choreography scores for the same song. "
    "You cannot hear the audio; the supplied acoustic summary is fixed-parameter measurement "
    "(relative energy, onset rate, energy trend per block) and its repeat/contrast entries are "
    "similarity guesses, not detected sections. "
    "Score each version 1-5 on four criteria: musical_fit (does movement amount follow the acoustic "
    "block evidence), motif_development (themes introduced/varied/returned on purpose, not mechanically "
    "copied), action_executability (does each action fit its stated duration and read as a doable movement), "
    "and boundary_connection (do adjacent events connect coherently). "
    "Sustained movement and intentional motif return are not defects. Unknown physical state is not a defect. "
    "Cite concrete event ids for every score. "
    'Return only {"A":{"musical_fit":n,"motif_development":n,"action_executability":n,'
    '"boundary_connection":n,"reason":"..."},"B":{...},"preferred":"A"|"B"|"tie"}.'
)


def call(base_url, model, messages, max_tokens=900, seed=7):
    body = json.dumps(dict(model=model, messages=messages, temperature=0, max_tokens=max_tokens,
                           seed=seed, response_format={"type": "json_object"})).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read())
    return payload["choices"][0]["message"]["content"]


def compact(events, limit=200, hide_motif=False):
    return [dict(event_id=e["event_id"], start=e.get("start"), end=e.get("end"), track=e.get("track"),
                 action=e["action"][:limit], motif=None if hide_motif else e.get("motif_id")) for e in events]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="review-pairs.public.json")
    parser.add_argument("--music-root", required=True, help="directory with <song>.json summaries")
    parser.add_argument("--base-url", default="http://127.0.0.1:8014/v1")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge-seed", type=int, default=7)
    parser.add_argument("--action-chars", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--reverse-events", action="store_true",
                        help="show each plan's events in reverse order (presentation robustness check)")
    parser.add_argument("--hide-motif", action="store_true",
                        help="drop motif ids from both versions (label confound control)")
    parser.add_argument("--repeat-swapped", action="store_true",
                        help="Judge each pair twice, second time with A/B swapped, to measure consistency")
    args = parser.parse_args()
    bundle = json.loads(Path(args.pairs).read_text())
    results = []
    for pair in bundle["pairs"]:
        if args.only and pair["pair_id"] not in args.only:
            continue
        summary_path = Path(args.music_root) / (pair["song_id"] + ".json")
        summary = json.loads(summary_path.read_text())
        evidence = dict(blocks=[{k: b[k] for k in ("block_id", "start_seconds", "end_seconds",
                                                  "relative_energy_db", "onset_rate_per_second", "energy_trend")}
                                for b in summary["blocks"]],
                        relations=summary["relations"][:12],
                        note="acoustic measurement only; not chorus/verse/phrase/beat detection")
        payload = dict(song_id=pair["song_id"], comparison=pair["comparison"],
                       acoustic_evidence=evidence, A=compact(pair["A"][::-1] if args.reverse_events else pair["A"], args.action_chars, args.hide_motif),
                       B=compact(pair["B"][::-1] if args.reverse_events else pair["B"], args.action_chars, args.hide_motif),
                       criteria=list(CRITERIA))
        try:
            raw = call(args.base_url, args.model,
                       [dict(role="system", content=SYSTEM),
                        dict(role="user", content=json.dumps(payload, ensure_ascii=False))],
                       seed=args.judge_seed, max_tokens=args.max_tokens)
            value = json.loads(raw)
            row = dict(pair_id=pair["pair_id"], song_id=pair["song_id"], comparison=pair["comparison"],
                       scores=value, raw=raw, error=None)
            if args.repeat_swapped:
                flipped = dict(payload, A=payload["B"], B=payload["A"])
                raw2 = call(args.base_url, args.model,
                            [dict(role="system", content=SYSTEM),
                             dict(role="user", content=json.dumps(flipped, ensure_ascii=False))],
                            seed=args.judge_seed, max_tokens=args.max_tokens)
                value2 = json.loads(raw2)
                # Map the swapped answer back to the original orientation.
                mapped = dict(A=value2["B"], B=value2["A"], preferred=value2.get("preferred"))
                row["scores_swapped"] = mapped
                # Unmapped response, kept so position bias can be measured in response space.
                row["scores_swapped_positions"] = value2
                deltas = []
                for criterion in CRITERIA:
                    try:
                        d1 = float(value["A"][criterion]) - float(value["B"][criterion])
                        d2 = float(mapped["A"][criterion]) - float(mapped["B"][criterion])
                    except (KeyError, TypeError, ValueError):
                        deltas.append(None)
                        continue
                    deltas.append(dict(original=d1, swapped=d2,
                                       consistent=(d1 == 0 and d2 == 0) or (d1 > 0) == (d2 > 0)))
                row["consistency"] = dict(zip(CRITERIA, deltas))
        except Exception as error:
            row = dict(pair_id=pair["pair_id"], song_id=pair["song_id"], comparison=pair["comparison"],
                       scores=None, raw=None, error=repr(error))
        results.append(row)
        print(row["pair_id"], row["comparison"], "ok" if row["error"] is None else row["error"][:60], flush=True)
    if args.only and Path(args.out).is_file():
        previous = json.loads(Path(args.out).read_text())
        rows = [r for r in previous["pairs"] if r["pair_id"] not in {x["pair_id"] for x in results}] + results
        rows.sort(key=lambda r: r["pair_id"])
    else:
        rows = results
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(schema="choreobench.model_pairwise.v1",
                                              judge_model=args.model, judge_seed=args.judge_seed, pairs=rows,
                                              note="Model-assisted review, not human; never fed back to generation."),
                                         ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(pairs=len(results), failed=sum(1 for r in results if r["error"]))))


if __name__ == "__main__":
    main()
