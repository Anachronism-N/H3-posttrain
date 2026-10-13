"""E7-v1 step 3 (on-trajectory version): reward-weighted self-distillation with anchor gating on group rollouts.

Per group (K branches sharing one prompt row, rewards r_k):
  - anchor = branch 0. Gate: only branches with r_k > r_0 + margin are eligible; if none, the group gives no gradient.
  - eligible branches get Gibbs weights softmax(r_k / tau); a winner is sampled from them.
  - loss: for the winner branch, pick `--blocks_per_sample` blocks and a random step j in each; feed the block's OWN
    recorded input at step j (same noise trajectory that produced the winner), previous blocks = winner's clean
    output, and regress the student's x0 prediction onto that block's final output. The target is determined by the
    trajectory (no averaging over fresh noise, which blurred the student in the first, failed version).
    Conditioned on the INTENDED prompt (full motion caption), audio and reference image.
Stays close to the init by LoRA init from E6-3b, small lr, few epochs (acts as the KL anchor).
torchrun --nproc_per_node 8 e7_train.py --gen <dir> --init <gen_lora_ema.pt> --out <dir> --round r
"""
import argparse, glob, json, math, os, random, sys, time

import torch
import torch.distributed as dist
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dmd"))
sys.path.insert(0, os.path.join(HERE, "..", "e6"))
import dmd_core as C  # noqa: E402
import causal  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--gen", nargs="+", required=True, help="one or more rollout dirs (groups_*.jsonl, rewards_*.jsonl)")
p.add_argument("--init", required=True)
p.add_argument("--out", required=True)
p.add_argument("--ckpt", default="/dev/shm/od/ckpt/S4-final/step-final.safetensors")
p.add_argument("--sigmas", default="1.0,0.9405940771102905,0.8333333134651184,0.6372548937797546")
p.add_argument("--block", type=int, default=4)
p.add_argument("--height", type=int, default=1280)
p.add_argument("--width", type=int, default=704)
p.add_argument("--num_frames", type=int, default=81)
p.add_argument("--lr", type=float, default=5e-6)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--tau", type=float, default=0.05)
p.add_argument("--margin", type=float, default=0.02)
p.add_argument("--blocks_per_sample", type=int, default=3)
p.add_argument("--max_grad_norm", type=float, default=1.0)
args = p.parse_args()
SIG = [float(x) for x in args.sigmas.split(",")]

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dev = f"cuda:{local}"
os.makedirs(args.out, exist_ok=True)


def log(m):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- load groups + rewards, gate
items = []
stats = dict(groups=0, gated=0, win_k=[0, 0, 0, 0], gain=[])
for d in args.gen:
    G = {}
    for f in glob.glob(f"{d}/groups_*.jsonl"):
        for l in open(f):
            g = json.loads(l); G[g["sid"]] = g
    for f in glob.glob(f"{d}/rewards_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["sid"] not in G:
                continue
            g = G[r["sid"]]; rs = {x["k"]: x["reward"] for x in r["res"]}
            stats["groups"] += 1
            elig = [k for k in rs if k != 0 and rs[k] > rs[0] + args.margin]
            if not elig:
                continue
            w = [math.exp(rs[k] / args.tau) for k in elig]
            z = sum(w)
            items.append(dict(g=g, dir=d, elig=elig, w=[x / z for x in w], anchor=rs[0], rs=rs))
            stats["gated"] += 1
log(f"groups {stats['groups']}, eligible (gate passed) {stats['gated']}")
if rank == 0:
    json.dump(dict(groups=stats["groups"], gated=stats["gated"], mean_anchor=sum(i["anchor"] for i in items) / max(1, len(items)),
                   mean_best=sum(max(i["rs"].values()) for i in items) / max(1, len(items))),
              open(os.path.join(args.out, "gate_stats.json"), "w"))
if not items:
    log("no eligible groups; nothing to train")
    dist.barrier()
    sys.exit(0)

time.sleep(local * 8)
pipe, _ = C.build_pipe(args.ckpt, dev)
pipe.dit.to(dev)
C.inject_dual_lora(pipe.dit, 128, 128)
C.load_lora_state(pipe.dit, torch.load(args.init, map_location="cpu"))
gen_params = C.lora_params(pipe.dit, "gen")
for q in gen_params:
    q.requires_grad_(True)
for q in C.lora_params(pipe.dit, "fake"):
    q.requires_grad_(False)
for q in gen_params:
    dist.broadcast(q.data, 0)
causal.enable((args.height // 32) * (args.width // 32), args.block)
T_LAT = (args.num_frames - 1) // 4 + 1
BOUNDS = causal.block_bounds(T_LAT, args.block)
opt = torch.optim.AdamW(gen_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
rng = random.Random(777 + rank)
order_rng = random.Random(5)
total_steps = args.epochs * (len(items) // world)
log(f"{len(items)} items, {args.epochs} epochs, {total_steps} steps")

step = 0
t_last = time.time()
for ep in range(args.epochs):
    idx = list(range(len(items)))
    order_rng.shuffle(idx)
    for s0 in range(0, len(idx) - world + 1, world):
        it = items[idx[s0 + rank]]
        g = it["g"]
        k = rng.choices(it["elig"], weights=it["w"])[0]
        d = torch.load(f"{it['dir']}/{g['sid']}_k{k}.pt", map_location="cpu")
        lat, traj = d["lat"].to(dev).to(torch.bfloat16), d["traj"]
        img = Image.open(g["image_path"]).convert("RGB").resize((args.width, args.height), Image.BICUBIC)
        ae = torch.load(g["mert_path"], map_location="cpu", weights_only=True)[..., :args.num_frames].to(dev, torch.bfloat16)
        shared, posi, _ = C.prepare_sample(pipe, g["prompt_full"], img, ae, args.height, args.width, args.num_frames, seed=rng.randint(0, 2 ** 31))
        opt.zero_grad(set_to_none=True)
        lsum = 0.0
        for bi in rng.sample(range(len(traj)), min(args.blocks_per_sample, len(traj))):
            tb = traj[bi]; s_, e_ = tb["s0"], tb["e0"]
            j = rng.randint(0, len(SIG) - 1)
            cur = torch.cat([lat[:, :, :s_], tb["xs"][j].to(dev, torch.bfloat16), torch.zeros_like(lat[:, :, e_:])], dim=2)
            sf = torch.ones(T_LAT, device=dev); sf[:s_] = 0.0; sf[s_:e_] = SIG[j]
            v = causal.velocity_pf(pipe, shared, posi, cur, sf, grad=True, mode="gen")
            pred = (cur - SIG[j] * v)[:, :, s_:e_]
            loss_b = ((pred.float() - tb["out"].to(dev).float()) ** 2).mean() / args.blocks_per_sample
            C.DualLoRALinear.mode = "gen"
            loss_b.backward()
            lsum += loss_b.item()
        loss = torch.tensor(lsum)
        flat = torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).flatten() for q in gen_params])
        dist.all_reduce(flat); flat /= world
        off = 0
        for q in gen_params:
            n = q.numel(); q.grad = flat[off:off + n].view_as(q).clone(); off += n
        gn = torch.nn.utils.clip_grad_norm_(gen_params, args.max_grad_norm)
        opt.step()
        step += 1
        if rank == 0:
            log(json.dumps(dict(step=step, ep=ep, loss=round(loss.item(), 4), gn=round(gn.item(), 4), win_k=k, sec=round(time.time() - t_last, 1))))
        t_last = time.time()
if rank == 0:
    torch.save(C.lora_state_dict(pipe.dit, "gen"), os.path.join(args.out, "gen_lora_ema.pt"))  # same name so downstream tools reuse it
    open(os.path.join(args.out, "DONE"), "w").write("ok")
dist.barrier()
log("finished")
