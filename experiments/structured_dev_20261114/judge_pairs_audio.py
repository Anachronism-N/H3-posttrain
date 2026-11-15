#!/usr/bin/env python3
"""Audio-grounded blind pairwise review with a different model family.

Unlike judge_pairs.py (text-only judge reading an acoustic summary), this judge
receives the full song audio and can hear energy changes and onsets directly.
It is still a model, not a human, and is never fed back into generation.
Each answer must first restate what it heard, so audio attention can be audited.
"""
import argparse
import base64
import hashlib
import json
import urllib.request
from pathlib import Path

CRITERIA = ("musical_fit", "motif_development", "action_executability", "boundary_connection")
SYSTEM = (
    "You can hear the supplied music. Compare two anonymous timed choreography scores written for it. "
    "Use the audio: movement density should follow audible energy and onset density, transitions should not "
    "land randomly against clear audible changes, and sustained passages may legitimately stay sustained. "
    "Score each version 1-5 on: musical_fit (does the amount and kind of movement match what you hear), "
    "motif_development (themes introduced, varied and returned on purpose rather than mechanically copied), "
    "action_executability (each action fits its stated duration and reads as a doable movement), "
    "boundary_connection (adjacent events connect coherently). Times are milliseconds. "
    "Sustained movement and intentional motif return are not defects. Cite concrete event ids. "
    'Return only {"heard":["...","..."],"A":{"musical_fit":n,"motif_development":n,"action_executability":n,'
    '"boundary_connection":n,"reason":"..."},"B":{...},"preferred":"A"|"B"|"tie"}. '
    '"heard" must be two short sentences about the music only, never about the scores.'
)


def audio_part(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


def call(base_url, model, system, payload, audio, max_tokens=1100, seed=7):
    body = json.dumps(dict(model=model, temperature=0, max_tokens=max_tokens, seed=seed,
                           response_format={"type": "json_object"},
                           messages=[dict(role="system", content=[dict(type="text", text=system)]),
                                     dict(role="user", content=[audio_part(audio),
                                                                dict(type="text", text=json.dumps(payload,
                                                                                                  ensure_ascii=False))])])).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())


def first_object(text):
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in: " + text[:200])
    depth, in_string, escape = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError("unterminated JSON object")


def compact(events, limit=180):
    return [dict(event_id=e["event_id"], start=e.get("start"), end=e.get("end"), track=e.get("track"),
                 action=e["action"][:limit], motif=e.get("motif_id")) for e in events]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="review-pairs.public.json")
    parser.add_argument("--audio-root", required=True, help="directory with <song_id>.wav at 16 kHz mono")
    parser.add_argument("--base-url", default="http://127.0.0.1:8016/v1")
    parser.add_argument("--model", default="Qwen2.5-Omni-7B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge-seed", type=int, default=11)
    parser.add_argument("--action-chars", type=int, default=180)
    parser.add_argument("--repeat-swapped", action="store_true")
    args = parser.parse_args()
    bundle = json.loads(Path(args.pairs).read_text())
    audio_cache = {}
    results = []
    for pair in bundle["pairs"]:
        audio = Path(args.audio_root) / (pair["song_id"] + ".wav")
        payload = dict(song_id=pair["song_id"], comparison=pair["comparison"],
                       A=compact(pair["A"], args.action_chars), B=compact(pair["B"], args.action_chars),
                       criteria=list(CRITERIA))
        try:
            raw_response = call(args.base_url, args.model, SYSTEM, payload, audio, seed=args.judge_seed)
            answer = raw_response["choices"][0]["message"]["content"]
            value = json.loads(first_object(answer))
            row = dict(pair_id=pair["pair_id"], song_id=pair["song_id"], comparison=pair["comparison"],
                       scores=value, usage=raw_response.get("usage"), raw=answer, error=None)
            if args.repeat_swapped:
                flipped = dict(payload, A=payload["B"], B=payload["A"])
                second = call(args.base_url, args.model, SYSTEM, flipped, audio, seed=args.judge_seed)
                answer2 = second["choices"][0]["message"]["content"]
                value2 = json.loads(first_object(answer2))
                mapped = dict(A=value2["B"], B=value2["A"], preferred=value2.get("preferred"),
                              heard=value2.get("heard"))
                row["scores_swapped"] = mapped
                row["raw_swapped"] = answer2
                check = []
                for criterion in CRITERIA:
                    try:
                        d1 = float(value["A"][criterion]) - float(value["B"][criterion])
                        d2 = float(mapped["A"][criterion]) - float(mapped["B"][criterion])
                    except (KeyError, TypeError, ValueError):
                        check.append(None)
                        continue
                    check.append(dict(original=d1, swapped=d2,
                                      consistent=(d1 == 0 and d2 == 0) or (d1 > 0) == (d2 > 0)))
                row["consistency"] = dict(zip(CRITERIA, check))
        except Exception as error:
            row = dict(pair_id=pair["pair_id"], song_id=pair["song_id"], comparison=pair["comparison"],
                       scores=None, raw=None, error=repr(error))
        audio_cache[pair["song_id"]] = hashlib.sha256(audio.read_bytes()).hexdigest()[:16]
        results.append(row)
        print(row["pair_id"], row["comparison"], "ok" if row["error"] is None else row["error"][:80], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(schema="choreobench.audio_pairwise.v1", judge_model=args.model,
                                              judge_seed=args.judge_seed, audio_sha16=audio_cache, pairs=results,
                                              note="Audio-grounded model review, not human; never fed back to generation."),
                                         ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(pairs=len(results), failed=sum(1 for r in results if r["error"]))))


if __name__ == "__main__":
    main()
