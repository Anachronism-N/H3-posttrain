"""cache_meta_list.py <data_process cache dir> <metadata.json> <out_meta.json> <out_list.tsv> [nproc=8 ranks]: for a raw data_process cache
(<dir>/<rank>/<i>.pth <-> metadata[(i*8+rank) % len]) write sid -> sizes (n_ra, n_img, n_aud, L) and a 'path\\tsid' list (wrapped padding dropped)."""
import glob, json, os, sys
from multiprocessing import Pool
import torch
P = 8
if sys.argv[1:2] == ["--validate_cfg_view"]:
    import hashlib
    root, files_out = sys.argv[2:4]
    r = "/opt/h3cache"
    files = sorted(glob.glob(f"{root}/*/*.pth"))
    assert files, "empty CFG training view"
    meta = json.load(open(sys.argv[4] if len(sys.argv) > 4 else "/dev/shm/h3data/metadata_f0_2700.json"))
    result = []
    for f in files:
        real = os.path.realpath(f)
        rank, index = int(os.path.basename(os.path.dirname(f))), int(os.path.basename(f)[:-4])
        sid = os.path.basename(meta[index * 8 + rank]["video"])[:-4]
        c = torch.load(real, map_location="cpu", weights_only=False, mmap=True)
        rel = os.path.relpath(real, r)[:-4]
        pos = torch.load(f"{r}/posi/{rel}.posi.pt", map_location="cpu", weights_only=False, mmap=True)
        neg = torch.load(f"{r}/nega/{rel}.nega.pt", map_location="cpu", weights_only=False, mmap=True)
        assert pos["Lp"] == neg["Lp"] < c[1]["prompt_embeds"].shape[0], sid
        assert tuple(c[0]["input_latents"].shape) == (1, 24, 37, 52, 30), sid
        assert c[0]["audio_input_latents"].shape[-1] == 207, sid
        pk_old = c[1]["packed"]
        for sc in (pos, neg):
            pk = sc["packed"]
            assert sc["Lp"] + sc["tail"].shape[0] == pk["text_pos"].numel(), sid
            assert sc["tail"].shape[1] == c[1]["prompt_embeds"].shape[1], sid
            for k in ("img_pos", "audio_pos"):
                assert pk[k].numel() == pk_old[k].numel(), (sid, k)
                assert int(pk[k].min()) >= 0 and int(pk[k].max()) < pk["seq_len"], (sid, k)
            for k in ("img_pos", "audio_pos"):
                new_pos = pk["img_position_ids"][0, pk[k]]
                old_pos = pk_old["img_position_ids"][0, pk_old[k]]
                delta = pk["text_pos"].numel() - pk_old["text_pos"].numel()
                assert torch.equal(new_pos[:, 1:], old_pos[:, 1:]), (sid, k, "spatial positions")
                assert torch.allclose(new_pos[:, 0], old_pos[:, 0] + delta, rtol=0, atol=1e-9), (sid, k, "text-offset positions")
                assert torch.equal(pk["token_tags"][pk[k]], pk_old["token_tags"][pk_old[k]]), (sid, k, "tags")
        h = hashlib.sha256()
        with open(real, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        result.append({"view": os.path.relpath(f, root), "sid": sid, "sha256": h.hexdigest()})
        if len(result) % 200 == 0:
            print("[cfg check]", len(result), flush=True)
    with open(files_out, "w") as stream:
        json.dump(result, stream)
    print("[cfg check] done", len(result), flush=True)
    sys.exit(0)

D, MD, OM, OL = sys.argv[1:5]
M = [os.path.basename(m["video"])[:-4] for m in json.load(open(MD))]
def one(f):
    r, i = int(os.path.basename(os.path.dirname(f))), int(os.path.basename(f)[:-4]); j = i * P + r
    if j >= len(M): return None
    c = torch.load(f, map_location="cpu", weights_only=False, mmap=True); pk = c[1]["packed"]
    return f, M[j], dict(n_ra=int(c[0]["ref_audio_anchor"].shape[0]), n_img=int(pk["img_pos"].numel()), n_aud=int(pk["audio_pos"].numel()), L=int(c[1]["prompt_embeds"].shape[0]))
if __name__ == "__main__":
    with Pool(16) as p:
        res = [x for x in p.map(one, sorted(glob.glob(f"{D}/*/*.pth")), chunksize=32) if x]
    json.dump({s: m for _, s, m in res}, open(OM, "w")); open(OL, "w").write("".join(f"{f}\t{s}\n" for f, s, _ in res)); print("[meta]", len(res))
