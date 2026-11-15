#!/usr/bin/env python3
"""Test-retest reliability of the text judge: same pairs, paraphrased rubric, new seed.

If the verdict survives a different wording and seed, the preference is less
likely to be an artifact of the original prompt. It is still one model family.
"""
import argparse
import json
from pathlib import Path

from judge_pairs import CRITERIA, call, compact
from judge_pairs_audio import first_object

SYSTEM = (
    "Two anonymous choreography scores were written for the same song. Work in this order: "
    "(1) for each score decide whether the number of simultaneous moves rises where the measured "
    "acoustic energy rises and falls where it falls; (2) check whether a repeated theme is varied "
    "rather than pasted unchanged; (3) check whether each action is physically plausible in its "
    "stated time span; (4) check whether neighbouring events join without a jump. "
    "The supplied block measurements are fixed-parameter acoustic data, not detected sections; "
    "you cannot hear the audio. Sustained movement is not a defect and returning a theme on purpose "
    "is not a defect. Give 1-5 for each of musical_fit, motif_development, action_executability, "
    "boundary_connection, quote the event ids that decided your rating, and then choose the score "
    "you would rather rehearse. "
    'Answer with only {"A":{...},"B":{...},"preferred":"A"|"B"|"tie"} using the four criterion keys.'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--music-root", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8014/v1")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge-seed", type=int, default=23)
    parser.add_argument("--action-chars", type=int, default=120)
    parser.add_argument("--only", nargs="*", help="restrict to these pair ids")
    args = parser.parse_args()
    bundle = json.loads(Path(args.pairs).read_text())
    rows = []
    for pair in bundle["pairs"]:
        if args.only and pair["pair_id"] not in args.only:
            continue
        summary = json.loads((Path(args.music_root) / (pair["song_id"] + ".json")).read_text())
        evidence = dict(blocks=[{k: b[k] for k in ("block_id", "start_seconds", "end_seconds",
                                                  "relative_energy_db", "onset_rate_per_second", "energy_trend")}
                                for b in summary["blocks"]],
                        relations=summary["relations"][:12],
                        note="acoustic measurement only")
        payload = dict(song_id=pair["song_id"], acoustic_evidence=evidence,
                       A=compact(pair["A"], args.action_chars), B=compact(pair["B"], args.action_chars),
                       criteria=list(CRITERIA))
        try:
            raw = call(args.base_url, args.model,
                       [dict(role="system", content=SYSTEM),
                        dict(role="user", content=json.dumps(payload, ensure_ascii=False))],
                       seed=args.judge_seed, max_tokens=1400)
            rows.append(dict(pair_id=pair["pair_id"], song_id=pair["song_id"],
                             comparison=pair["comparison"], scores=json.loads(first_object(raw)), raw=raw, error=None))
        except Exception as error:
            rows.append(dict(pair_id=pair["pair_id"], song_id=pair["song_id"],
                             comparison=pair["comparison"], scores=None, raw=None, error=repr(error)))
        print(rows[-1]["pair_id"], "ok" if rows[-1]["error"] is None else rows[-1]["error"][:70], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.only and Path(args.out).is_file():
        previous = json.loads(Path(args.out).read_text())
        keep = {r["pair_id"]: r for r in previous["pairs"] if r["pair_id"] not in {x["pair_id"] for x in rows}}
        rows = sorted(list(keep.values()) + rows, key=lambda r: r["pair_id"])
    Path(args.out).write_text(json.dumps(dict(schema="choreobench.model_pairwise.v1", judge_model=args.model,
                                              judge_seed=args.judge_seed, rubric="paraphrased_v2", pairs=rows,
                                              note="Test-retest replicate; not human."), ensure_ascii=False,
                                         indent=1) + "\n")


if __name__ == "__main__":
    main()
