#!/usr/bin/env python3
"""Per-GPU matmul holder: 100% duty cycle, ~0.4GB memory. Usage: OCCUPY_GPU=<n> python3 hold_gpu.py"""
import os, time
import torch

gpu = int(os.environ["OCCUPY_GPU"])
torch.cuda.set_device(gpu)
dev = f"cuda:{gpu}"
a = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
b = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
print(f"[holder] gpu={gpu} holding", flush=True)
while True:
    for _ in range(50):
        c = a @ b
    torch.cuda.synchronize(gpu)
    time.sleep(0.0)
