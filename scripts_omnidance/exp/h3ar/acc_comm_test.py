import os, time, torch, accelerate
t = time.time()
acc = accelerate.Accelerator()
x = torch.ones(1, device=acc.device) * acc.process_index
import torch.distributed as dist
dist.all_reduce(x)
print(f"[acc] rank {acc.process_index}/{acc.num_processes} sum={x.item()} init {time.time()-t:.1f}s", flush=True)
