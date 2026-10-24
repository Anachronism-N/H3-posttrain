"""Build a local read-only review bundle from completed annotations, not inference."""
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path("/apdcephfs_gy4/share_302533218/cedricnie")
MS = ROOT / "diff/logs_omnidance/exp/MS"
OUT = ROOT / "diff/logs_omnidance/exp/review_web/annotation_compare_v1"
SKELETON_IDS = ("wJPptuhclZmKAEWs", "wJPrpFsThWAcBQIw", "wJQTclrlOLEYYfTB")
CHUNK_IDS = ("hwrlCvOCYUKpljlV", "LglrebUOmlpWTFrr", "qKgmlBaDdOeqvUZZ")
BODY_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
EXTRA_EDGES = ((20,22),(20,23),(22,23),(21,24),(21,25),(24,25),
               (7,28),(28,26),(28,27),(26,27),(8,31),(31,29),(31,30),(29,30))


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def checked_report(path):
    record, ref = load(path)
    receipt, _ = load(Path(path).with_suffix(".receipt.json"))
    if receipt["report_sha256"] != ref["sha256"]:
        raise ValueError("Caption source report receipt changed")
    return record, ref


def skeleton(sid):
    bundle = MS / "nlf_rgb32_fresh_detector_fixed3_v1/raw" / sid
    manifest, ref = load(bundle / "manifest.json")
    receipt, _ = load(bundle / "manifest.receipt.json")
    if receipt["manifest_sha256"] != ref["sha256"]:
        raise ValueError("Fresh32 manifest receipt changed")
    raw = (bundle / "raw.npz").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["raw_sha256"]:
        raise ValueError("Raw point bundle changed")
    if manifest["source"] != "nlf_fresh_detector_rgb_landmarks":
        raise ValueError("Only fresh detector32 allowed")
    with np.load(bundle / "raw.npz", allow_pickle=False) as archive:
        points = archive["points2d_px"][32]
        valid = archive["valid"][32]
        names = archive["names"].tolist()
        timestamp = float(archive["timestamps"][32])
    image = None
    with av.open(manifest["source_paths"]["video"]) as source:
        for index, frame in enumerate(source.decode(video=0)):
            if index == 32:
                image = frame.to_ndarray(format="bgr24")
                break
    if image is None:
        raise ValueError("Actual source frame32 absent")
    height, width = image.shape[:2]
    support = valid & np.isfinite(points).all(-1) & (points[:,0]>=0) & (points[:,0]<width) & (points[:,1]>=0) & (points[:,1]<height)
    overlay = image.copy()
    edges = [(p,i) for i,p in enumerate(BODY_PARENTS) if p>=0] + list(EXTRA_EDGES)
    for a,b in edges:
        if support[a] and support[b]:
            color = (80,210,80) if max(a,b)<22 else (255,155,30) if max(a,b) in (22,23,26,27,28) else (30,105,255)
            cv2.line(overlay, tuple(points[a].astype(int)), tuple(points[b].astype(int)), color, 2)
    for i,p in enumerate(points):
        if support[i]:
            color = (80,210,80) if i<22 else (255,155,30) if i in (22,23,26,27,28) else (30,105,255)
            cv2.circle(overlay,tuple(p.astype(int)),4 if i<22 else 6,color,-1)
            if i>=22:cv2.putText(overlay,str(i),tuple(p.astype(int)+[7,-7]),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
    footer = np.zeros((78,width*2,3),np.uint8)
    cv2.putText(footer,"RGB | Fresh32 native2D: green body, blue left extras, orange right extras",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    cv2.putText(footer,f"frame32 PTS {timestamp:.4f}; in-frame support only, NOT calibrated visibility / GT",(10,58),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    combined = np.vstack((np.hstack((image,overlay)),footer))
    path = OUT / (sid + ".skeleton.png")
    if not cv2.imwrite(str(path),combined):raise RuntimeError("Image encoding failed")
    return {"id":sid,"preview":str(path),"video":str(MS / "nlf_rgb32_fresh_detector_fixed3_v1/comparison" / sid / "original_cached32_fresh_detector32.mp4"),
        "original_video":manifest["source_paths"]["video"],"frame_index":32,"time":timestamp,
        "extra_points":[{"index":i,"name":names[i],"in_frame_support":bool(support[i])} for i in range(22,32)],
        "raw_manifest":ref,"raw_sha256":manifest["raw_sha256"],"visibility_verified":False}


def ms_view(label, views):
    return {"label":label,"source":"MotionScript","windows":[{
        "start":v["window_start"],"end":v["window_end"],"status":v["status"],
        "reason":v.get("abstain_reason"),"caption":v["cleaned_motion_text"],
        "partial_caption":v.get("partial_supported_motion_text",""),
        "raw_original_MS":[{key:part.get(key) for key in ("start","end","ms","ms_nonagg")} for part in v["original_captions"]]} for v in views]}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary, vlm_ref = load(MS / "visual_music_windows_first3_run_v1/summary.json")
    if not summary["all_ready_windows_terminal"] or not summary["all_plans_executed"]:
        raise ValueError("Shared VLM plans not fully executed")
    if summary["window_count"] != 22 or summary["stage"] != "run":raise ValueError("Shared VLM trial not terminal")
    chunks=[]
    for sid in CHUNK_IDS:
        base, ref = checked_report(MS / "ms_beat_C_first50_v1" / (sid + ".json"))
        views=[ms_view("A 整段",[base["A_whole_clip"]]),ms_view("B 固定约2秒",base["B_fixed_windows"]),ms_view("C 检测拍点",base["C_windows"])]
        refs=[ref]
        for label,folder,tag in (("BPM 长度窗","ms_bpm_duration_first50_v1","bpm"),
                                ("小节 strict","ms_truebar_first50_strict_v1","bar_strict"),
                                ("小节 merged短边缘","ms_truebar_first50_merge_short_edges_v1","bar_merge_short_edges")):
            record, ref = checked_report(MS / folder / (sid + ".json"));refs.append(ref)
            views.append(ms_view(label,record["MS_windows"]))
            visual=next(r for r in summary["records"] if r["id"]==sid and r["tag"]==tag)
            if visual["shared_plan"] != record["window_manifest"]:
                raise ValueError("MS and VLM shared source/window plan mismatch")
            bounds = [(w["window"]["start"], w["window"]["end"]) for w in visual["windows"]]
            if bounds != [(w["window_start"], w["window_end"]) for w in record["MS_windows"]]:
                raise ValueError("MS and VLM observed different windows")
            views.append({"label":label,"source":"独立VLM RGB","windows":[{
                "start":w["window"]["start"],"end":w["window"]["end"],"status":w["status"],
                "caption":w["caption"],"events":w["events"]} for w in visual["windows"]]})
        chunks.append({"id":sid,"video":base["manifest_record"]["video_path"],"audio":base["manifest_record"]["audio_path"],
            "views":views,"source_refs":refs+[vlm_ref],"audio_video_alignment_verified":False})
    result={"schema":"annotation_compare_review_bundle_v1","skeleton_cases":[skeleton(sid) for sid in SKELETON_IDS],
        "caption_cases":chunks,"note":"骨架三case与分窗三case是两组不同源，不冒同源增点分窗联合对照。原MS/VLM英文正文保留，不重写、补空或批准语义；窗时间不是精确动作边界。",
        "model_inference_repeated":False,"reviews_or_scores_modified":False,"semantic_approval":False}
    path=OUT / "data.json"
    temp=OUT / "data.pending.json"
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+chr(10))
    temp.replace(path)
    print(json.dumps({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"skeleton_cases":3,"caption_cases":3,"model_inference_repeated":False}))


if __name__=="__main__":main()
