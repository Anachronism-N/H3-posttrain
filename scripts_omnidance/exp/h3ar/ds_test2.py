import os, time, torch, torch.distributed as dist
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
import deepspeed
t = time.time(); deepspeed.init_distributed(dist_backend="nccl")
x = torch.ones(1, device="cuda"); dist.all_reduce(x)
print(f"[ds2] rank {dist.get_rank()} sum {x.item()} {time.time()-t:.1f}s", flush=True)
