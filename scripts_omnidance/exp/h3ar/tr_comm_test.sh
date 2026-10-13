#!/bin/bash
NR=$1; M=$2
PATH=/opt/venv_h3/bin:$PATH timeout 300 /opt/venv_h3/bin/python -m torch.distributed.run --nnodes 3 --node_rank $NR --nproc_per_node 8 --master_addr $M --master_port 29891 -c "import deepspeed" 2>/dev/null
cat > /tmp/ds_test.py <<'PY'
import os, time, torch, deepspeed, torch.distributed as dist
t=time.time(); deepspeed.init_distributed(dist_backend="nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); x=torch.ones(1,device="cuda"); dist.all_reduce(x)
print(f"[ds] rank {dist.get_rank()} sum {x.item()} {time.time()-t:.1f}s", flush=True)
PY
PATH=/opt/venv_h3/bin:$PATH timeout 300 /opt/venv_h3/bin/python -m torch.distributed.run --nnodes 3 --node_rank $NR --nproc_per_node 8 --master_addr $M --master_port 29892 /tmp/ds_test.py
