"""Real frames/music + explicitly illustrative bar-based choreography text.

Run: python figures/source/plot_choreography_example_v2.py
All media and detections are hash-checked. No models or network are used.
"""
from pathlib import Path
import hashlib
import json
import re
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
import numpy as np


SOURCE = Path(__file__).resolve().parent
OUT = SOURCE.parent
CASE = SOURCE / "case_example"
CASE_ID = "wJPptuhclZmKAEWs"
SIZE = (6.875, 3.20)
NAVY, TEAL, INDIGO, OCHRE = "#26445C", "#238B8E", "#65518B", "#946B1D"
INK, MUTED, GRID = "#263746", "#61717C", "#DDE5EA"
TEXTS = [
    "At chest height, sweep both hands side to side in compact arcs, keeping the elbows softly bent.",
    "Then lower both hands together toward the waist while the torso gently turns; keep the movement range compact.",
    "Lift both forearms toward chest height, then let the hands travel across the torso; retain bent elbows.",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_inputs():
    annotation_file = CASE / (CASE_ID + ".audit.json")
    frame_file = CASE / (CASE_ID + ".frame_audit.json")
    beat_file = CASE / (CASE_ID + ".beat_audit.json")
    annotation = json.loads(annotation_file.read_text(encoding="utf-8"))
    frame = json.loads(frame_file.read_text(encoding="utf-8"))
    beat = json.loads(beat_file.read_text(encoding="utf-8"))
    assert annotation["example_id"] == frame["case_id"] == beat["case_id"] == CASE_ID
    assert annotation["caption"]["dance_style"] == "contemporary"
    assert sha(annotation_file) == beat["paired_annotation_source"]["audit_sha256"]
    wav = CASE / (CASE_ID + ".wav")
    audio_sha = sha(wav)
    assert audio_sha == annotation["audio"]["sha256"] == frame["audio_sha256"]
    assert audio_sha == beat["raw_prediction_record"]["audio_sha256"]
    assert audio_sha == beat["input_preflight_record"]["audio_sha256"]
    assert audio_sha == beat["local_audio_verification"]["sha256"]
    assert beat["local_audio_verification"]["matches_original_inference_audio"] is True
    assert beat["raw_prediction_record"]["downbeats"] == [0.2, 1.96, 3.7]
    assert beat["bar_interpretation"]["complete_predicted_bar_intervals_seconds"] == [[0.2, 1.96], [1.96, 3.7]]
    assert beat["inference_configuration"]["dbn"] is False
    assert beat["model"]["checkpoint_variant"] == "final0"
    assert sha(CASE / (CASE_ID + ".mp4")) == frame["video_sha256"] == annotation["video"]["sha256"]
    assert frame["video_probe"]["fps_numerator"] == 16
    assert frame["video_probe"]["num_frames"] == 81
    for row in frame["frames"]:
        assert sha(CASE / row["file"].replace("\\", "/")) == row["sha256"]
        assert row["time_seconds"] == row["zero_based_video_frame"] / 16
    assert [r["time_seconds"] for r in frame["frames"]] == [0.5, 1.5, 2.5, 3.5, 4.5]
    with wave.open(str(wav), "rb") as stream:
        sr, n = stream.getframerate(), stream.getnframes()
        assert sr == 24000 and n == 121500
        assert stream.getnchannels() == 1 and stream.getsampwidth() == 2
        audio = np.frombuffer(stream.readframes(n), dtype="<i2").astype(float) / 32768
    assert n / sr == 5.0625 == beat["raw_prediction_record"]["duration_seconds"]
    sources = {
        "annotation_audit": dict(file=str(annotation_file.relative_to(SOURCE)), sha256=sha(annotation_file)),
        "frame_audit": dict(file=str(frame_file.relative_to(SOURCE)), sha256=sha(frame_file)),
        "beat_audit": dict(file=str(beat_file.relative_to(SOURCE)), sha256=sha(beat_file)),
        "audio": dict(file=str(wav.relative_to(SOURCE)), sha256=audio_sha),
    }
    return annotation, frame, beat, audio, sr, sources


def rich_paragraph(fig, renderer, text, x, top, width, fontsize=8.3):
    """Preserve text while wrapping by real glyph widths in the final renderer."""
    body = {"both", "hands", "elbows", "torso", "forearms"}
    spatial = {"chest", "height", "compact", "waist", "toward"}
    relation = {"then", "while", "keeping"}
    max_width = fig.bbox.width * width
    rows, current, current_width = [], [], 0
    for token in text.split():
        key = re.sub(r"[^a-z]", "", token.lower())
        color = TEAL if key in body else INDIGO if key in spatial else OCHRE if key in relation else INK
        prop = dict(fontfamily="STIXGeneral", fontsize=fontsize, color=color,
                    fontweight="bold" if key in body | spatial | relation else "normal")
        if key in spatial:
            prop["bbox"] = dict(boxstyle="round,pad=0.025", facecolor="#EEEAF6", edgecolor="none")
        word = TextArea(token + " ", textprops=prop)
        word.set_figure(fig)
        length = word.get_window_extent(renderer).width
        if current and current_width + length > max_width:
            rows.append(HPacker(children=current, align="baseline", pad=0, sep=0))
            current, current_width = [], 0
        current.append(word)
        current_width += length
    if current:
        rows.append(HPacker(children=current, align="baseline", pad=0, sep=0))
    block = VPacker(children=rows, align="left", pad=0, sep=1.5)
    annotation = AnnotationBbox(block, (x, top), xycoords=fig.transFigure,
                               box_alignment=(0, 1), frameon=False, pad=0)
    fig.add_artist(annotation)
    return annotation, len(rows)


def main():
    annotation, frame, beat, audio, sr, sources = checked_inputs()
    plt.rcParams.update({
        "font.family": "STIXGeneral", "font.size": 8.3,
        "text.color": INK, "axes.labelcolor": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "path.simplify": True, "path.simplify_threshold": 0.02,
    })
    fig = plt.figure(figsize=SIZE, dpi=240)
    fig.text(0.04, 0.985, "Dance style is contemporary", fontsize=10.4,
             color=NAVY, fontweight="bold", va="top")
    fig.text(0.96, 0.985, "Source style metadata  |  illustrative rewrite",
             fontsize=8.3, color=MUTED, ha="right", va="top")

    left, right, duration = 0.04, 0.96, 5.0625
    tx = lambda t: left + (right-left) * t / duration
    photo_height = 0.35
    photo_width = photo_height * SIZE[1] / SIZE[0] * 704 / 1280
    for row in frame["frames"]:
        t = row["time_seconds"]
        ax = fig.add_axes([tx(t)-photo_width/2, 0.57, photo_width, photo_height])
        ax.imshow(plt.imread(CASE / row["file"].replace("\\", "/")), interpolation="antialiased")
        ax.set(xticks=[], yticks=[])
        color = TEAL if t < 1.96 else INDIGO if t < 3.70 else OCHRE
        for edge in ax.spines.values():
            edge.set_color(color)
            edge.set_linewidth(0.8)
        fig.text(tx(t), 0.548, f"{t:.1f} s", fontsize=8.3, color=NAVY,
                 ha="center", va="center")

    ax = fig.add_axes([left, 0.425, right-left, 0.09])
    ax.axvspan(0, 0.2, color="#ADB7BF", alpha=0.12, lw=0)
    ax.axvspan(0.2, 1.96, color=TEAL, alpha=0.055, lw=0)
    ax.axvspan(1.96, 3.7, color=INDIGO, alpha=0.045, lw=0)
    ax.axvspan(3.7, duration, color=OCHRE, alpha=0.055, lw=0)
    time = np.arange(len(audio), dtype=float) / sr
    ax.plot(time, audio, color="#8798A4", linewidth=0.16, alpha=0.14, zorder=1)
    width = int(round(0.020 * sr))
    starts = np.arange(0, len(audio), width)
    ends = np.minimum(starts+width, len(audio))
    centers = (starts+ends)/(2*sr)
    rms = np.array([np.sqrt(np.mean(audio[a:b]**2)) for a,b in zip(starts, ends)])
    ax.plot(centers, rms, color=TEAL, linewidth=1.0, zorder=3)
    ax.plot(centers, -rms, color=TEAL, linewidth=1.0, zorder=3)
    for db in beat["raw_prediction_record"]["downbeats"]:
        ax.axvline(db, color=OCHRE, lw=0.8, linestyle=(0, (3, 2)), zorder=4)
    peak = np.abs(audio).max()
    ax.set(xlim=(0, duration), ylim=(-peak*1.05, peak*1.05), xticks=[], yticks=[])
    for edge in ax.spines.values():
        edge.set_visible(False)
    fig.text(left, 0.400, "20 ms RMS (teal) + raw waveform (gray)",
             fontsize=8.1, color=MUTED, va="center")
    fig.text(right, 0.400, "Dashed: predicted downbeats; lead-in: 0–0.20 s",
             fontsize=8.1, color=MUTED, va="center", ha="right")

    # One two-bar unit with a merged lead-in; the suffix is an explicit fragment.
    bx0, bx1, by = left, tx(3.7), 0.345
    fig.add_artist(plt.Line2D([bx0,bx0,bx1,bx1], [by-.012,by,by,by-.012],
                             transform=fig.transFigure, color=NAVY, linewidth=.8))
    fig.text((bx0+bx1)/2, 0.350, "2-bar unit (lead-in merged): 0–3.70 s",
             fontsize=8.4, color=NAVY, ha="center", va="bottom", fontweight="bold")

    cards = [
        dict(x=left, width=tx(1.96)-left-.010, title="Bar 1 [0.20, 1.96] s", color=TEAL),
        dict(x=tx(1.96)+.004, width=tx(3.7)-tx(1.96)-.014, title="Bar 2 [1.96, 3.70] s", color=INDIGO),
        dict(x=tx(3.7)+.004, width=right-tx(3.7)-.004, title="Residual [3.70, 5.0625] s", color=OCHRE),
    ]
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    line_counts, text_artists = [], []
    for card, text in zip(cards, TEXTS):
        rect = FancyBboxPatch((card["x"],.105),card["width"],.215,
                              boxstyle="round,pad=0.004",transform=fig.transFigure,
                              edgecolor=card["color"],facecolor="white",linewidth=.65,zorder=-1)
        fig.add_artist(rect)
        fig.text(card["x"]+.008,.303,card["title"],fontsize=8.4,
                 fontweight="bold",color=card["color"],va="top")
        artist, count = rich_paragraph(fig, renderer, text, card["x"]+.008, .250,
                                       card["width"]-.016, fontsize=8.3)
        text_artists.append(artist)
        line_counts.append(count)

    badges = [("D1  Specific actions/range",TEAL),
              ("D2  Clear body/space",INDIGO),
              ("D3  Order/coordination",OCHRE)]
    for i,(label,color) in enumerate(badges):
        fig.text(0.04+i*.313,.040,label,fontsize=8.2,color=color,
                 fontweight="bold",va="center",
                 bbox=dict(boxstyle="round,pad=.2",facecolor="#F7F9FA",edgecolor=GRID,lw=.5))
    fig.canvas.draw()
    fonts=[t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    assert min(fonts)>=8.1
    for artist in text_artists:
        assert artist.get_window_extent(fig.canvas.get_renderer()).y0 > fig.bbox.height*.085, "Text overflows card"
    outputs={}
    for ext in ("pdf","svg","png"):
        path=OUT/f"choreography_example_v2.{ext}"
        fig.savefig(path,dpi=360,bbox_inches=None,pad_inches=0,
                    metadata={"Creator":"Matplotlib; real media and audited Beat This predictions"})
        outputs[path.name]=sha(path)
    plt.close(fig)
    provenance={
        "schema":"herewechoreography.real_case_figure.v2","case_id":CASE_ID,
        "figure_size_inches":list(SIZE),"minimum_font_points":min(fonts),
        "source_files":sources,"source_frame_files":frame["frames"],
        "style":"contemporary","style_basis":"source annotation metadata, not human validation",
        "predicted_downbeats_seconds":beat["raw_prediction_record"]["downbeats"],
        "detector_checkpoint_sha256":beat["model"]["checkpoint_sha256"],
        "detector_source_evidence_sha256":beat["source_evidence"]["sha256"],
        "complete_predicted_bars_seconds":[[.2,1.96],[1.96,3.70]],
        "prefix_fragment_seconds":[0,.2],"suffix_fragment_seconds":[3.7,duration],
        "planning_units_seconds":[[0,3.70],[3.70,duration]],
        "first_unit_type":"Two complete predicted bars with merged lead-in; not two pure bars",
        "rms":{"sample_rate":sr,"window_samples":width,"nominal_window_ms":20,
               "last_bin_samples":int(ends[-1]-starts[-1]),"time_seconds":centers.tolist(),
               "amplitudes":rms.tolist(),"pcm_normalization":"signed int16 / 32768",
               "drawing":"positive and negative RMS lines; no solid amplitude fill"},
        "illustrative_rewrite":dict(origin="Author-composed example based on true source frames; not original GT or model output",
             text_cards=TEXTS,wrapped_lines=line_counts,quality_validated=False,exact_motion_timing_validated=False),
        "highlight_legend":{"teal":"body parts","indigo_with_background":"range/spatial wording","ochre":"language relations"},
        "rubric_badges":"Explanatory D1-D3 criteria, not numeric scores or human approval",
        "video_modified_with_ai":False,"pose_skeleton_generated":False,"new_model_inference":False,
        "beat_accuracy_or_audio_video_alignment_verified":False,
        "music_fit_or_choreography_quality_claimed":False,"output_sha256":outputs,
    }
    target=SOURCE/"choreography_example_v2.audit.json"
    target.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(outputs=list(outputs),size_inches=list(SIZE),text_lines=line_counts,
                         minimum_font_points=min(fonts)),indent=2))


if __name__=="__main__":
    main()
