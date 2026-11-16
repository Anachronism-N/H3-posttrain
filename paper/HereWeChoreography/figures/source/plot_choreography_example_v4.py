"""Editable qualitative score: measured audio, real frames, author illustration.

Run: python figures/source/plot_choreography_example_v4.py
Frames are decoded without AI modification. Display cropping removes background
only; all source pixels, times and hashes remain auditable. No model is run.
"""
import json
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
import numpy as np

from plot_choreography_example_v2 import (
    CASE, CASE_ID, SOURCE, OUT, checked_inputs, sha,
)

SIZE = (6.875, 3.55)
NAVY, TEAL, PURPLE, GOLD = "#223E56", "#007F83", "#68518E", "#9C6A12"
INK, MUTED, LINE = "#273844", "#667780", "#DAE3E8"
FRAME_INDICES = [0, 20, 36, 56, 68, 80]
DISPLAY_CROP = (120, 280, 640, 1250)
TEXTS = [
    "Lower the raised foot; sweep both hands side to side in a small range in front of the upper torso, with elbows softly bent.",
    "Then lower both hands toward the waist while keeping the elbows softly bent.",
    "From the waist, lift both forearms in front of the upper torso; near the end, raise one knee.",
]
ROLES = {
    "body": (TEAL, ["the raised foot", "both hands", "both forearms", "elbows", "one knee"]),
    "action": (NAVY, ["lower", "sweep", "lift", "raise"]),
    "space": (PURPLE, ["small range", "in front of the upper torso", "toward the waist", "from the waist"]),
    "order": (GOLD, ["then", "while", "near the end"]),
}


def rich_text(fig, renderer, text, x, top, width, fontsize=8.3):
    """Role-consistent phrase highlighting, wrapped by actual glyph widths."""
    phrases = {phrase: (role, color) for role, (color, words) in ROLES.items() for phrase in words}
    pattern = re.compile("(" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + ")", re.I)
    spans = [(match.start(), match.end(), phrases[match.group().lower()])
             for match in pattern.finditer(text)]
    tokens = []
    matches = list(re.finditer(r"\S+\s*", text))
    assert "".join(match.group() for match in matches) == text
    for match in matches:
        role, color = next((info for a,b,info in spans if a <= match.start() < b), (None, INK))
        props = dict(fontfamily="STIXGeneral", fontsize=fontsize, color=color,
                     fontweight="bold" if role else "normal")
        if role in ("space", "body"):
            props["bbox"] = dict(boxstyle="round,pad=.035", facecolor="#EFEBF6" if role == "space" else "#E7F3F1", edgecolor="none")
        word = TextArea(match.group(), textprops=props)
        word.set_figure(fig)
        tokens.append(word)
    rows, current, length = [], [], 0
    for token in tokens:
        w = token.get_window_extent(renderer).width
        if current and length + w > fig.bbox.width * width:
            rows.append(HPacker(children=current, align="baseline", pad=0, sep=0))
            current, length = [], 0
        current.append(token)
        length += w
    if current:
        rows.append(HPacker(children=current, align="baseline", pad=0, sep=0))
    block = VPacker(children=rows, align="left", pad=0, sep=1.0)
    artist = AnnotationBbox(block, (x, top), xycoords=fig.transFigure,
                            box_alignment=(0, 1), frameon=False, pad=0)
    fig.add_artist(artist)
    return artist, len(rows)


def rectangle(fig, x, y, width, height, fill, edge=LINE, radius=.008, lw=.65):
    patch = FancyBboxPatch((x, y), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        transform=fig.transFigure, facecolor=fill, edgecolor=edge, lw=lw, zorder=-2)
    fig.add_artist(patch)


def frames():
    target = CASE / (CASE_ID + "_frames_v4")
    target.mkdir(exist_ok=True)
    expected = [target / f"frame_{i + 1:02}.png" for i in range(len(FRAME_INDICES))]
    if not all(path.is_file() for path in expected):
        expr = "+".join(f"eq(n,{index})" for index in FRAME_INDICES)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(CASE / (CASE_ID + ".mp4")), "-vf", f"select='{expr}'",
                        "-fps_mode", "passthrough", str(target / "frame_%02d.png")],
                       check=True, capture_output=True, text=True)
    assert len(list(target.glob("frame_*.png"))) == len(FRAME_INDICES)
    return [dict(file=str(path.relative_to(SOURCE)), sha256=sha(path),
                 video_frame=index, seconds=index / 16)
            for path, index in zip(expected, FRAME_INDICES)]


def main():
    annotation, old_frames, beat, audio, sr, sources = checked_inputs()
    selected = frames()
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 8.2,
        "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig = plt.figure(figsize=SIZE, dpi=240)
    left, right, duration = .034, .966, len(audio) / sr
    tx = lambda t: left + (right - left) * t / duration
    boundary = tx(3.70)
    fig.text(left, .985, "Dance style is contemporary", color=NAVY,
             fontsize=10.5, fontweight="bold", va="top")
    fig.text(right, .985, "Real media  |  illustrative score", color=MUTED,
             fontsize=8.1, ha="right", va="top")
    for x, role, label in [(.034, "body", "Body"), (.154, "action", "Action"),
                            (.290, "space", "Space / range"), (.505, "order", "Order / coordination")]:
        fig.text(x, .937, label, color=ROLES[role][0], fontsize=8.1,
                 fontweight="bold", va="top")

    # Measured extrema preserve dynamics. One global scale, no local gain,
    # invented musical accents, or replacement with a decorative waveform.
    ax = fig.add_axes([left, .837, right - left, .060])
    step = int(round(.002 * sr))
    starts = np.arange(0, len(audio), step)
    ends = np.minimum(starts + step, len(audio))
    centers = (starts + ends) / (2 * sr)
    low = np.array([audio[a:b].min() for a, b in zip(starts, ends)])
    high = np.array([audio[a:b].max() for a, b in zip(starts, ends)])
    for a, b, color in [(0, 3.7, TEAL), (3.7, duration, GOLD)]:
        ax.axvspan(a, b, color=color, alpha=.035, lw=0)
    ax.fill_between(centers, low, high, color=TEAL, alpha=.77, lw=0)
    for value in beat["raw_prediction_record"]["downbeats"]:
        ax.axvline(value, color=GOLD, lw=.7, ls=(0, (2, 2)))
    peak = np.abs(audio).max()
    ax.set(xlim=(0, duration), ylim=(-peak * 1.03, peak * 1.03), xticks=[], yticks=[])
    for edge in ax.spines.values():
        edge.set_visible(False)
    for a, b, label, color in [(.2, 1.96, "Bar 1  ·  0.20–1.96 s", TEAL),
                              (1.96, 3.7, "Bar 2  ·  1.96–3.70 s", TEAL),
                              (3.7, duration, "Fragment  ·  3.70–5.06 s", GOLD)]:
        fig.text((tx(a)+tx(b))/2, .810, label, fontsize=8.1,
                 color=color, fontweight="bold", ha="center", va="center")

    # The same display-only viewport keeps head, hands and feet in every frame.
    crop_x0, crop_y0, crop_x1, crop_y1 = DISPLAY_CROP
    photo_h = .285
    photo_w = photo_h * SIZE[1] / SIZE[0] * (crop_x1-crop_x0)/(crop_y1-crop_y0)
    for i, row in enumerate(selected):
        x = float(np.clip(tx(row["seconds"]), left + photo_w/2, right - photo_w/2))
        image_ax = fig.add_axes([x-photo_w/2, .477, photo_w, photo_h])
        pixels = plt.imread(SOURCE / row["file"])
        assert pixels.shape[:2] == (1280, 704)
        image_ax.imshow(pixels, interpolation="antialiased")
        image_ax.set(xlim=(crop_x0, crop_x1), ylim=(crop_y1, crop_y0), xticks=[], yticks=[])
        for edge in image_ax.spines.values():
            edge.set_color("#C7D4DA"); edge.set_linewidth(.55)
        fig.text(x, .459, f"{chr(65+i)}  {row['seconds']:.2f} s", fontsize=8.1,
                 color=NAVY, ha="center", va="center")

    # Two top-level units, with two unboxed phases inside the first one.
    gap = .018
    unit_x, unit_w = left, boundary - left - gap/2
    tail_x, tail_w = boundary + gap/2, right - boundary - gap/2
    rectangle(fig, unit_x, .168, unit_w, .261, "#F5FAFA", "#B8DAD7")
    rectangle(fig, tail_x, .168, tail_w, .261, "#FCF9F2", "#E6D4AB")
    fig.text(unit_x+.012, .418, "Two-bar planning unit  |  0–3.70 s", fontsize=8.9,
             color=TEAL, fontweight="bold", va="top")
    fig.text(tail_x+.011, .418, "Ending  |  3.70–5.06 s", fontsize=8.9,
             color=GOLD, fontweight="bold", va="top")
    split = unit_x + unit_w * .53
    fig.add_artist(plt.Line2D([split, split], [.188, .369],
                             transform=fig.transFigure, color="#D6E6E5", lw=.6))
    phase_boxes = [(unit_x+.012, split-unit_x-.025, "A–C  ·  Settle and sweep"),
                   (split+.012, unit_x+unit_w-split-.024, "C–D  ·  Lower together"),
                   (tail_x+.011, tail_w-.022, "D–F  ·  Lift, then raise knee")]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_artists, line_counts = [], []
    for text, (x, width, title) in zip(TEXTS, phase_boxes):
        fig.text(x, .372, title, fontsize=8.1, color=NAVY,
                 fontweight="bold", va="top")
        artist, count = rich_text(fig, renderer, text, x, .335, width, fontsize=8.3)
        text_artists.append(artist); line_counts.append(count)

    # This line reads the written instructions; it is not a measured pose trace.
    fig.text(left, .133, "TEXTUAL HAND-OFF", fontsize=8.1,
             color=NAVY, fontweight="bold", va="center")
    fig.text(.267, .133, "hands near waist", fontsize=8.2, color=TEAL,
             fontweight="bold", va="center")
    fig.add_artist(FancyArrowPatch((.475, .133), (.532, .133),
        transform=fig.transFigure, arrowstyle="->", mutation_scale=7, lw=.9, color=NAVY))
    fig.text(.545, .133, "lift from that position", fontsize=8.2,
             color=PURPLE, fontweight="bold", va="center")
    fig.add_artist(plt.Line2D([left, right], [.100, .100],
                             transform=fig.transFigure, color=LINE, lw=.65))
    for x, title, detail, color in [
        (left, "Specific  ·  D1/D2", "Body + motion + spatial range", TEAL),
        (.365, "Ordered  ·  D3", "Sequence + coordination", GOLD),
        (.694, "Connected  ·  L1", "End state supports the next action", PURPLE)]:
        fig.text(x, .080, title, fontsize=8.3, color=color, fontweight="bold", va="top")
        fig.text(x, .043, detail, fontsize=8.1, color=INK, va="top")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fonts = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    assert min(fonts) >= 8.1
    for artist in text_artists:
        assert artist.get_window_extent(renderer).y0 > fig.bbox.height * .178, "Instruction exceeds card"
    outputs = {}
    for ext in ("pdf", "svg", "png"):
        path = OUT / f"choreography_example_v4.{ext}"
        fig.savefig(path, dpi=360, bbox_inches=None, pad_inches=0,
                    metadata={"Creator": "Matplotlib; unchanged real media and author illustration"})
        outputs[path.name] = sha(path)
    plt.close(fig)
    provenance = dict(schema="herewechoreography.real_case_figure.v4", case_id=CASE_ID,
        figure_size_inches=SIZE, minimum_font_points=min(fonts), source_files=sources,
        selected_frames=selected, source_video_sha256=old_frames["video_sha256"],
        frame_extraction="FFmpeg select by zero-based source index; passthrough; lossless PNG",
        display_viewport_xyxy=DISPLAY_CROP, display_crop_scope="Background only; all six figures retain full body",
        waveform=dict(kind="2-ms measured PCM min/max envelope", global_scaling_only=True,
                      sample_rate=sr, sample_count=len(audio), synthetic_or_locally_amplified=False),
        predicted_downbeats_seconds=beat["raw_prediction_record"]["downbeats"],
        detector_checkpoint_sha256=beat["model"]["checkpoint_sha256"],
        planning_units_seconds=[[0,3.7],[3.7,duration]],
        complete_bars_seconds=[[.2,1.96],[1.96,3.7]], merged_lead_in_seconds=[0,.2],
        ending_is_incomplete_fragment=True,
        text_origin="Author-composed illustrative score, not GT or model output", text_cards=TEXTS,
        text_line_counts=line_counts, phase_frame_references=["A-C", "C-D", "D-F"],
        phase_timing="Illustrative internal phases, not annotated exact movement boundaries",
        hand_off="Textually specified hand state, not measured body-state continuity",
        source_style_metadata="contemporary", video_modified_with_ai=False,
        new_model_inference=False, independent_av_alignment_verified=False,
        music_fit_or_quality_measured=False, quality_ratings=None,
        output_sha256=outputs)
    (SOURCE / "choreography_example_v4.audit.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(outputs=list(outputs), text_lines=line_counts,
                         size_inches=SIZE, minimum_font_points=min(fonts)), indent=2))


if __name__ == "__main__":
    main()
