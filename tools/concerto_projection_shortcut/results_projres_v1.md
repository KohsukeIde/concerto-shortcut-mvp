# ProjRes v1 Results

Updated: 2026-04-16 04:35 JST

## Bottom Line

`projres_v1a` completed the H100 continuation, ARKit stress, and ScanNet
linear gate. The ScanNet gate is a no-go for that arm.

`projres_v1b` then factorized the fix into partial target residualization
(`beta`) and a prediction-side coordinate alignment penalty (`alpha`). This
improved substantially over `projres_v1a` and over `no-enc2d-renorm`, but still
did not match the original Concerto continuation on the ScanNet linear proxy.

Best v1b arm:
- arm: `combo-b075-a001`
- beta: `0.75`
- alpha: `0.01`
- ScanNet linear last/best mIoU: `0.4220` / `0.4220`
- delta vs original: `-0.0574` / `-0.0332`
- delta vs `no-enc2d-renorm`: `+0.0426` / `+0.0418`

Decision:
- `projres_v1a`: no-go
- `projres_v1b`: no strong-go
- do not launch optional fine-tune yet
- next method should not be another full-removal projection residual; the
  useful region is partial coordinate removal around `beta=0.75`, but it still
  needs a better objective to close the remaining gap to original.

## ProjRes v1b Factorized Ablation

### Implementation

Compared with `projres_v1a`, `projres_v1b` adds
`coord_projection_beta`:

```text
u = normalize(stopgrad(g(c)))
t_res = t0 - beta * dot(t0, u) * u
loss = 1 - cos(y0, t_res) + alpha * cos(y0, u)^2
```

New logging metrics:
- `coord_removed_energy`
- `coord_projection_loss_check`

The metric sanity job for the old v1a setting (`beta=1.0`, `alpha=0.05`) passed
with `coord_projection_loss_check=0.0`, so the current code path is
algebraically consistent.

### Smoke Matrix

Run:
- summary root:
  `data/runs/projres_v1b/summaries/h10016-qf1-v1b-pre256`
- launcher:
  `tools/concerto_projection_shortcut/launch_projres_v1b_smoke_matrix.sh`
- resource:
  11 independent `rt_QF=1` jobs
- bounded steps:
  `CONCERTO_MAX_TRAIN_ITER=256`
- walltime requested:
  `00:35:00`
- jobs:
  `132209.qjcm` to `132219.qjcm`, all completed

All 11 smoke arms passed and all reported `coord_projection_loss_check=0.0`.

Top arms selected for continuation:

| rank | arm | beta | alpha | smoke score | last enc2d | last residual enc2d | last alignment | last pred energy | last residual norm |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | combo-b075-a001 | 0.75 | 0.01 | 9.9456 | 9.9456 | 9.6058 | 0.0139 | 0.0014 | 0.8963 |
| 2 | penalty-b000-a002 | 0.00 | 0.02 | 9.9624 | 9.8049 | 9.3888 | 0.0927 | 0.0093 | 0.9512 |
| 3 | resonly-b075-a000 | 0.75 | 0.00 | 10.1531 | 9.8906 | 9.5262 | 0.1346 | 0.0135 | 0.8934 |
| 4 | combo-b050-a002 | 0.50 | 0.02 | 10.1989 | 9.8339 | 9.3499 | 0.1761 | 0.0176 | 0.8932 |

### Continuation

Run:
- summary root:
  `data/runs/projres_v1b/summaries/h10016x4-qf16`
- resource:
  four continuation jobs, each `rt_QF=4` (4 nodes / 16 H100 GPUs)
- total concurrent allocation:
  16 nodes / 64 H100 GPUs
- epoch count:
  `CONCERTO_EPOCH=5`
- walltime requested:
  `01:35:00`

Jobs:

| job | arm | beta | alpha | status | walltime | checkpoint |
| --- | --- | ---: | ---: | --- | --- | --- |
| 132220.qjcm | combo-b075-a001 | 0.75 | 0.01 | Exit 0 | 00:46:37 | `exp/concerto/arkit-full-projres-v1b-combo-b075-a001-h10016x4-qf16-continue/model/model_last.pth` |
| 132221.qjcm | penalty-b000-a002 | 0.00 | 0.02 | Exit 0 | 00:46:33 | `exp/concerto/arkit-full-projres-v1b-penalty-b000-a002-h10016x4-qf16-continue/model/model_last.pth` |
| 132222.qjcm | resonly-b075-a000 | 0.75 | 0.00 | Exit 0 | 00:47:13 | `exp/concerto/arkit-full-projres-v1b-resonly-b075-a000-h10016x4-qf16-continue/model/model_last.pth` |
| 132223.qjcm | combo-b050-a002 | 0.50 | 0.02 | Exit 0 | 00:47:16 | `exp/concerto/arkit-full-projres-v1b-combo-b050-a002-h10016x4-qf16-continue/model/model_last.pth` |

Final epoch train result:

| arm | loss | enc2d | residual enc2d | alignment | target energy | removed energy | pred energy | residual norm | loss check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| combo-b075-a001 | 7.8700 | 7.6640 | 7.2912 | 1.7203 | 0.1305 | 0.0734 | 0.1720 | 0.8921 | 0.0000 |
| penalty-b000-a002 | 7.1661 | 6.1307 | 5.6151 | 6.8981 | 0.1297 | 0.0000 | 0.6898 | 0.9540 | 0.0000 |
| resonly-b075-a000 | 7.8755 | 7.6564 | 7.2862 | 1.9807 | 0.1300 | 0.0731 | 0.1981 | 0.8906 | 0.0000 |
| combo-b050-a002 | 7.6891 | 7.2467 | 6.8132 | 3.9045 | 0.1300 | 0.0325 | 0.3904 | 0.9059 | 0.0000 |

### ARKit Stress

Source root:
`data/runs/projres_v1b/summaries/h10016x4-qf16`

Enc2d loss mean over 20 batches:

| arm | clean | local surface destroy | z flip | xy swap | roll 90 x |
| --- | ---: | ---: | ---: | ---: | ---: |
| combo-b075-a001 | 7.649344 | 8.862813 | 8.860726 | 7.673076 | 9.093753 |
| penalty-b000-a002 | 6.382515 | 7.147604 | 7.189544 | 6.387743 | 7.172994 |
| resonly-b075-a000 | 7.682956 | 8.633449 | 8.549670 | 7.691244 | 8.754042 |
| combo-b050-a002 | 7.441266 | 8.376788 | 8.274656 | 7.421181 | 8.421834 |

### ScanNet Linear Gate

Source root:
`data/runs/projres_v1b/summaries/h10016x4-qf16`

| arm | beta | alpha | last mIoU | best mIoU | delta last vs original | delta best vs original | delta last vs no-enc2d-renorm | delta best vs no-enc2d-renorm | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| combo-b075-a001 | 0.75 | 0.01 | 0.4220 | 0.4220 | -0.0574 | -0.0332 | +0.0426 | +0.0418 | no strong-go |
| resonly-b075-a000 | 0.75 | 0.00 | 0.4176 | 0.4176 | -0.0618 | -0.0376 | +0.0382 | +0.0374 | no strong-go |
| combo-b050-a002 | 0.50 | 0.02 | 0.4129 | 0.4129 | -0.0665 | -0.0423 | +0.0335 | +0.0327 | no strong-go |
| penalty-b000-a002 | 0.00 | 0.02 | 0.3887 | 0.3907 | -0.0907 | -0.0645 | +0.0093 | +0.0105 | no strong-go |

Readout:
- partial target residualization is better than full removal.
- `beta=0.75` is the best tested region.
- penalty-only is weakest among the continued v1b arms.
- v1b does not justify optional fine-tuning yet because no arm reaches the
  original continuation gate.

## ProjRes v1a Full Removal

### Run

| item | value |
| --- | --- |
| selected alpha | `0.05` |
| selected prior | `mlp` |
| continuation experiment | `arkit-full-projres-v1a-alpha005-h10032-qf32-continue` |
| continuation job | `132196.qjcm`, `Exit_status=0`, walltime `00:39:37` |
| follow-up job | `132198.qjcm`, `Exit_status=0`, walltime `00:50:06` |
| continuation checkpoint | `exp/concerto/arkit-full-projres-v1a-alpha005-h10032-qf32-continue/model/model_last.pth` |
| linear experiment | `scannet-proxy-projres-v1a-alpha005-h10032-qf32-lin` |

### Continuation

Final epoch train result:

| loss | enc2d loss | residual enc2d loss | alignment | target energy | pred energy | residual norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8.0899 | 8.0655 | 6.3309 | 0.0200 | 0.1061 | 0.0020 | 0.7308 |

### ARKit Stress

Source:
`data/runs/projres_v1/summaries/h10032-qf32/arkit-full-projres-v1a-alpha005-h10032-qf32-continue_stress.csv`

| stress | batches | enc2d loss mean |
| --- | ---: | ---: |
| clean | 20 | 8.022868 |
| local_surface_destroy | 20 | 9.115467 |
| z_flip | 20 | 8.941781 |
| xy_swap | 20 | 8.022733 |
| roll_90_x | 20 | 9.353544 |

### ScanNet Linear Gate

Source:
`data/runs/projres_v1/summaries/h10032-qf32/scannet-proxy-projres-v1a-alpha005-h10032-qf32-lin_gate.json`

| arm | last mIoU | best mIoU |
| --- | ---: | ---: |
| ProjRes v1a | 0.3627 | 0.3627 |
| original continuation | 0.4794 | 0.4552 |
| no-enc2d-renorm | 0.3794 | 0.3802 |

Gate:

| comparison | last mIoU delta | best mIoU delta |
| --- | ---: | ---: |
| ProjRes v1a - original | -0.1167 | -0.0925 |
| ProjRes v1a - no-enc2d-renorm | -0.0167 | -0.0175 |

Decision:
- `strong_go=false`
- reason: `linear_gate_not_strong_go`
- do not launch the optional fine-tune from this arm under the current gate.
