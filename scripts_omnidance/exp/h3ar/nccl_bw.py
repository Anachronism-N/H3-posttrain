import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl"); r = dist.get_rank(); w = dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
for mb in (64, 512, 2048):
    x = torch.ones(mb * 2 ** 20 // 2, dtype=torch.bfloat16, device="cuda")
    for _ in range(3): dist.all_reduce(x)
    torch.cuda.synchronize(); t = time.time(); n = 10
    for _ in range(n): dist.all_reduce(x)
    torch.cuda.synchronize(); dt = (time.time() - t) / n
    bus = 2 * (w - 1) / w * x.numel() * 2 / dt / 1e9
    if r == 0: print(f"[nccl] world={w} size={mb}MB time={dt*1000:.1f}ms busbw={bus:.1f}GB/s", flush=True)
dist.destroy_process_group()
