#!/usr/bin/env python3
"""Build a source-bound MuQ-JEV style-prior sidecar for one catalog.

Every 5.0625 s window of every song in the catalog contract is scored by the
frozen three-head JEV ensemble. The output validates against
choreobench.public_style_prior and is bound to the catalog's dataset_sha256.

Window audio hashes are the SHA256 of the exact window file handed to the
encoder (16 kHz mono WAV); the harness records but does not re-verify them.
"""
import argparse
import hashlib
import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

TRAINER = Path("/ssd/2/niezichao.nzc/sound-caption/experiments/m2d_planner/trainer")
sys.path.insert(0, str(TRAINER.parents[1]))
sys.path.insert(0, str(Path("/ssd/2/niezichao.nzc/sound-caption")))
sys.path.insert(0, "/ssd/2/niezichao.nzc/sound-caption/h3-posttrain/H3-posttrain-choreobench")

WINDOW = 81000  # 5.0625 s at 16 kHz
CLASSES = ("pop / casual", "hip-hop", "jazz funk / heels", "chinese classical",
           "chinese folk", "contemporary", "k-pop")


def windows(num_samples):
    starts = list(range(0, num_samples - WINDOW + 1, WINDOW))
    if num_samples >= WINDOW and starts[-1] != num_samples - WINDOW:
        starts.append(num_samples - WINDOW)
    return [(start, start + WINDOW) for start in starts]


def write_window(pcm, rate, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--model", default="/ssd/2/niezichao.nzc/artifacts/jev_style_prior/muq-jev-7class-v1")
    parser.add_argument("--model-config",
                        default="/ssd/2/niezichao.nzc/artifacts/jev_style_prior/muq-jev-7class-v1/encoder_config.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scratch", default="/tmp/jev_windows")
    args = parser.parse_args()
    from experiments.m2d_planner.trainer import music_genre_jev as j
    from experiments.m2d_planner.trainer import music_genre_jev_ensemble as ensemble
    from choreobench.public_style_prior import MODEL_IDENTITY, SCHEMA

    catalog = Path(args.catalog).resolve()
    contract = json.loads((catalog / "contract.json").read_text())
    dataset = contract["dataset_snapshot"]
    dataset_sha = contract["dataset_sha256"]
    config = ensemble.read_config(args.model_config)
    head, metadata = ensemble.load(args.model)
    encoder = j.load_encoder(config)
    if list(metadata["classorder"]) != list(CLASSES):
        raise SystemExit("class order drift")
    scratch = Path(args.scratch)
    songs = []
    for song in dataset["songs"]:
        audio_path = catalog / song["audio"]["path"]
        rate = song["audio"]["sample_rate"]
        pcm = None
        if not audio_path.is_file():
            # Catalogs may keep only the harness windows; concatenating them is
            # lossless because stride equals window length.
            parts = sorted((catalog / "audio" / song["song_id"]).glob("*.wav"))
            if not parts:
                raise SystemExit("no audio for " + song["song_id"])
            spans = windows(song["audio"]["num_samples"])
            if len(parts) != len(spans):
                raise SystemExit("window count drift for %s: %d vs %d" % (song["song_id"], len(parts), len(spans)))
            pcm = np.zeros(song["audio"]["num_samples"], dtype="<i2")
            for part, (start, end) in zip(parts, spans):
                with wave.open(str(part), "rb") as handle:
                    if handle.getframerate() != rate:
                        raise SystemExit("window rate drift for " + song["song_id"])
                    chunk = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
                if len(chunk) != end - start:
                    raise SystemExit("window length drift for " + song["song_id"])
                pcm[start:end] = chunk
        else:
            with wave.open(str(audio_path), "rb") as handle:
                rate = handle.getframerate()
                pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        if len(pcm) != song["audio"]["num_samples"]:
            raise SystemExit("audio identity drift for %s: %d vs %d" % (song["song_id"], len(pcm), song["audio"]["num_samples"]))
        rows = []
        window_dir = catalog / "audio" / song["song_id"]
        for index, (start, end) in enumerate(windows(len(pcm))):
            chunk = pcm[start:end]
            existing = window_dir / ("%04d.wav" % index)
            path = existing if (existing.is_file() and existing.stat().st_size == 44 + 2 * len(chunk)) else scratch / song["song_id"] / ("%04d.wav" % index)
            if path != existing:
                write_window(chunk, rate, path)
            source = dict(audio_path=str(path), audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          audio_duration_seconds=len(chunk) / rate,
                          audio_window_seconds=[0.0, len(chunk) / rate])
            feature = j.audio_features(encoder, config, source)
            _, predictions = j.predict_features(head, feature[None], metadata["classorder"], metadata["temperature"])
            probabilities = predictions[0]["probabilities"]
            rows.append(dict(start=start, end=end, audio_sha256=source["audio_sha256"],
                             probabilities=[float(p) for p in probabilities]))
        songs.append(dict(song_id=song["song_id"], audio_sha256=song["audio"]["sha256"], sample_rate=rate,
                          num_samples=len(pcm), windows=rows))
        print(song["song_id"], len(rows), "windows", flush=True)
    document = dict(schema=SCHEMA, dataset_sha256=dataset_sha,
                    model_identity=json.loads(json.dumps(MODEL_IDENTITY)), songs=songs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(songs=len(songs), windows=sum(len(s["windows"]) for s in songs), out=str(out))))


if __name__ == "__main__":
    main()
