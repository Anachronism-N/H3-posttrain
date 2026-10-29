# H3-posttrain

MiniMax-H3 post-training for music-driven dance video (OmniDance): bidirectional LoRA SFT (DeCFG adapters,
guidance loss, Hybrid BF16 base), AR/causal experiments, music-condition injection, evaluation and review tooling.

## Layout
- `scripts_omnidance/exp/h3ar/` — H3 training / sampling / caching: `train_ha2.sh`, `q2_20261010.sh` (job launcher),
  `sample_h3_ar.py` (bi / block-causal AR sampler), `h3_causal.py`, `text_recache.py` / `nega_cache.py` (rich-prompt and
  unconditional sidecars), `fuse_adapter.py` (DeCFG / ostris adapter fusion, Hybrid merge), `h3_music.py` (MERT / MoE),
  `h3_dpo.py`, `eval_sched.sh` + `eval_jobs*.txt` (evaluation queue).
- `scripts_omnidance/exp/h3/` — H3 zero-shot baseline (E3) and OOS T2VA.
- `scripts_omnidance/exp/review_web/` — human-review web page (`python3 server.py --port 8765`).
- `scripts_omnidance/exp/{e6,e7,e9,e10,e11,dmd,g2,eval}` — earlier Wan2.2 experiments and metrics.
- `notes/` — design notes and experiment logs (19: H3 plan/log, 20: music condition ideas, 21: CFG / quality research).
- `diffsynth_patch/` — changes on top of DiffSynth-Studio commit `974cfa37f27ac55eba3b6d10efa21f876900572d`
  (`git apply DiffSynth-Studio_974cfa3.patch`); `files/` holds the full patched files
  (loss.py guidance modes, runner.py step cap / music lr, H3 DiT, H3 pipeline, dataset loader sidecars, train.py).

Paths, node addresses and dataset locations are those of the original cluster and must be adapted.
Access tokens were removed (`<HF_TOKEN_REDACTED>`); supply your own via environment variables.
Model weights, caches, data and generated videos are not included.

## Music planner training

MOSS / Qwen Captioner training with configurable natural-language captions, an optional
input-only JEV-inspired dance-style head, and 32/64-GPU multi-node plans is documented in
[the training guide](notes/planner_training/MOSS_QWEN_64GPU_JEV_20261027.md).
Cluster plans and CPU tests do not constitute real A800 training or quality acceptance.
For MOSS with a discrete dance style and timestamped actions, use the
[v2 protocol and reviewed generation path](notes/planner_training/MOSS_STYLE_TIMED_REVIEW_20261028.md).
For single-node MOSS baseline, broader native audio LoRA, and frozen MusicFM fusion,
see [encoder experiments and launch commands](notes/planner_training/MOSS_ENCODER_EXPERIMENTS_20261029.md).
