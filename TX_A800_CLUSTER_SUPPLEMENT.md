# TX A800 Cluster Supplement

This branch is supplementary material produced on the TX A800 cluster
(8 nodes x 8x A800-80GB, Taiji platform). It contains:

- pipeline/annotation_runtime/ - production annotation pipeline code
  (original MotionScript unified generation, DWPose 2D keypoints + hand ROI,
  two-stage draft pools, dynamic ready pools, VLM motion revision, native
  resource manager with 160GiB admission gate, locks/leases lifecycle).
- pipeline/diff_scripts_ms/ - MotionScript v2 and caption scripts from the
  diff repo (previously untracked).
- pipeline/external_a100/ - worker code shipped with the external A100
  cluster result packets.
- Pending content of local commit be62c90 (cross-source VLM batching and
  supported-fact caption fast path) plus uncommitted notes/tests/results
  from the motionscript-optimization working tree.

Hardware/execution context, failure records and limitations are documented
under notes/. Data, weights, media, credentials and runtime state
(claims/locks/leases) are intentionally not included.
