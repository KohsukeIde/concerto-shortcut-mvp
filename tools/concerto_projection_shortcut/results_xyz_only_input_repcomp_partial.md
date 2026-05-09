# XYZ-only input representation companion partial results

These rows evaluate a Concerto/PTv3 representation companion in which the
point-side input is restricted to XYZ-derived geometry by zeroing non-XYZ input
features, while the continuation target remains the original cross-modal target.

The current rows are **partial-continuation diagnostics**. The continuation jobs
produced `model_last.pth` checkpoints, but the runs later hit OOM around epoch 2
of the planned 5-epoch continuation. Linear proxy jobs completed from the
available checkpoints. Do not report these rows as clean 5-epoch continuation
results.

## ScanNet linear proxy

| continuation seed | linear seed | mIoU (%) | mAcc (%) | allAcc (%) | best val mIoU (%) | status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 32.71 | 45.97 | 71.00 | 30.71 | partial continuation |
| 1 | 1 | 32.91 | 44.95 | 71.14 | 30.67 | partial continuation |
| 1 | 2 | 32.59 | 45.28 | 70.90 | 30.45 | partial continuation |
| 0 | 0 | 32.43 | 45.33 | 70.75 | 30.37 | partial continuation |
| 2 | 0 | 29.94 | 42.38 | 69.74 | 27.99 | partial continuation |
| 3 | 0 | 31.39 | 43.79 | 70.25 | 29.25 | partial continuation |

## Aggregates

- Same continuation checkpoint, three linear seeds: `32.74 +/- 0.16` mIoU.
- Four continuation seeds with linear seed 0: `31.62 +/- 1.25` mIoU.

## Interpretation guardrail

Use these as a fast feasibility check for the stronger xyz-only-input
representation-level diagnostic. A clean version should rerun continuation to
completion with the linear watcher gated on a completion marker rather than the
first appearance of `model_last.pth`.
