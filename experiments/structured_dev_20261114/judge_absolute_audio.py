#!/usr/bin/env python3
"""Audio-grounded absolute scoring, one blinded plan per call.

Pairwise audio judging is position-biased for this model family: it prefers the
second version regardless of content. Absolute scoring removes the comparison
from the prompt entirely, so the same plan is rated identically wherever it
appears. Every plan is scored blind; the caller maps positions to methods.
"""
import argparse
import json
from pathlib import Path

from judge_pairs_audio import CRITERIA, audio_part, call, first_object

SYSTEM = (
    "You can hear the supplied music. Rate one anonymous timed choreography score written for it. "
    "Use the audio: movement density should follow audible energy and onset density, transitions should not "
    "land randomly against clear audible changes, and sustained passages may legitimately stay sustained. "
    "Score 1-5 on: musical_fit (does the amount and kind of movement match what you hear), "
    "motif_development (themes introduced, varied and returned on purpose rather than mechanically copied), "
    "action_executability (each action fits its stated duration and reads as a doable movement), "
    "boundary_connection (adjacent events connect coherently). Times are milliseconds. "
    "Sustained movement and intentional motif return are not defects. Cite at least two event ids. "
    'Return only {"heard":["...","..."],"scores":{"musical_fit":n,"motif_development":n,'
    '"action_executability":n,"boundary_connection":n},"reason":"..."}. '
    '"heard" must be two short sentences about the music only, never about the score.'
)


def compact(events, limit=180):
    return [dict(event_id=e["event_id"], start=e.get("start"), end=e.get("end"), track=e.get("track"),
                 action=e["action"][:limit], motif=e.get("motif_id")) for e in events]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8017/v1")
    parser.add_argument("--model", default="MOSS-Audio-8B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge-seed", type=int, default=11)
    parser.add_argument("--action-chars", type=int, default=180)
    args = parser.parse_args()
    bundle = json.loads(Path(args.pairs).read_text())
    rows = []
    for pair in bundle["pairs"]:
        audio = Path(args.audio_root) / (pair["song_id"] + ".wav")
        row = dict(pair_id=pair["pair_id"], song_id=pair["song_id"], comparison=pair["comparison"], versions=[])
        for position in ("A", "B"):
            payload = dict(song_id=pair["song_id"], version=compact(pair[position], args.action_chars),
                           criteria=list(CRITERIA))
            try:
                response = call(args.base_url, args.model, SYSTEM, payload, audio, seed=args.judge_seed)
                answer = response["choices"][0]["message"]["content"]
                value = json.loads(first_object(answer))
                rows_ok = all(isinstance(value["scores"].get(c), int) for c in CRITERIA)
                if not rows_ok:
                    raise ValueError("missing criteria in " + json.dumps(value["scores"]))
                row["versions"].append(dict(position=position, scores=value["scores"],
                                            heard=value.get("heard"), reason=value.get("reason"),
                                            usage=response.get("usage"), raw=answer, error=None))
            except Exception as error:
                row["versions"].append(dict(position=position, scores=None, error=repr(error)))
        rows.append(row)
        status = "ok" if all(v["error"] is None for v in row["versions"]) else "FAIL"
        print(row["pair_id"], row["comparison"], status, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(schema="choreobench.absolute_audio.v1", judge_model=args.model,
                                              judge_seed=args.judge_seed, pairs=rows,
                                              note="Absolute audio-grounded scoring, one plan per call; "
                                                   "position cannot bias the score."),
                                         ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(pairs=len(rows), failed=sum(1 for r in rows if any(v["error"] for v in r["versions"])))))


if __name__ == "__main__":
    main()
