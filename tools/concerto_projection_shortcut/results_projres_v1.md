# ProjRes v1 Results

Updated: 2026-04-16 08:25 JST

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

`projres_v1c` tested the next hypothesis from the v1b readout: keep
`beta=0.75`, use low-capacity / height-biased priors, and avoid another broad
beta/alpha sweep. It also completed prior fitting, smoke, 5-epoch continuation,
ARKit stress, and ScanNet linear gates.

Best v1c arm:
- arm: `mlpz-b075-a001`
- prior: `mlp_z`
- beta: `0.75`
- alpha: `0.01`
- ScanNet linear last/best mIoU: `0.4186` / `0.4186`
- delta vs original: `-0.0608` / `-0.0366`
- delta vs `no-enc2d-renorm`: `+0.0392` / `+0.0384`

Decision:
- `projres_v1a`: no-go
- `projres_v1b`: no strong-go
- `projres_v1c`: no strong-go
- do not launch optional fine-tune yet
- v1c does not beat v1b, so changing only the static coordinate prior family is
  not enough. The next method should move beyond static projection residual
  removal and use a more selective or objective-level intervention.

## ProjRes v1c Selective Prior Family Ablation

### Implementation

`projres_v1c` keeps the v1b loss and adds lower-capacity coordinate prior
families:

```text
u = normalize(stopgrad(g(c)))
t_res = t0 - beta * dot(t0, u) * u
loss = 1 - cos(y0, t_res) + alpha * cos(y0, u)^2
```

Supported prior arch names:
- `linear` / `linear_xyz`: 3D xyz linear prior
- `mlp` / `mlp_xyz`: 3D xyz MLP prior
- `linear_z`: z-only linear prior
- `mlp_z`: z-only MLP prior

New / updated helpers:
- `fit_coord_prior.py` supports `--cache-root` and `--prior-archs`
- `submit_projres_v1c_fit_priors_abciq_qf.sh`
- `launch_projres_v1c_prior_matrix.sh`
- `launch_projres_v1c_continue_top.sh`
- `launch_projres_v1c_followup_from_manifest.sh`
- `summarize_projres_smoke_manifest.py`

### Prior Fit

Run:
- output root: `data/runs/projres_v1c/priors`
- cache reused: `data/runs/projres_v1/priors/cache`
- job: `132277.qjcm`, `rt_QF=1`, `Exit_status=0`
- walltime: requested `00:25:00`

| prior | val cosine loss | target energy | residual norm | checkpoint |
| --- | ---: | ---: | ---: | --- |
| linear_z | 0.735445 | 0.080087 | 0.958630 | `data/runs/projres_v1c/priors/linear_z/model_last.pth` |
| mlp_z | 0.643186 | 0.136156 | 0.928692 | `data/runs/projres_v1c/priors/mlp_z/model_last.pth` |

The `linear_xyz` arm reuses the existing v1 linear xyz prior:
`data/runs/projres_v1/priors/linear/model_last.pth`.

### Smoke Matrix

Run:
- summary root:
  `data/runs/projres_v1c/summaries/h10016-qf1-v1c-prior256`
- launcher:
  `tools/concerto_projection_shortcut/launch_projres_v1c_prior_matrix.sh`
- resource:
  six independent `rt_QF=1` jobs
- requested steps:
  `CONCERTO_MAX_TRAIN_ITER=256`
- walltime requested:
  `00:35:00`
- jobs:
  `132278.qjcm` to `132283.qjcm`

The 256-step smoke walltime was too tight; logs reached 190 to 193 steps. The
partial logs were summarized with a 128-step minimum and used only as a smoke
stability filter.

Top arms selected for continuation:

| rank | arm | prior | beta | alpha | smoke score | last enc2d | last residual enc2d | last pred energy | last residual norm | steps |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | linz-b075-a000 | linear_z | 0.75 | 0.00 | 9.9440 | 9.9440 | 9.9440 | 0.0007 | 0.9559 | 193 |
| 2 | mlpz-b075-a001 | mlp_z | 0.75 | 0.01 | 9.9885 | 9.9885 | 9.9885 | 0.0005 | 0.9273 | 193 |
| 3 | linxyz-b075-a001 | linear_xyz | 0.75 | 0.01 | 10.0232 | 10.0232 | 10.0231 | 0.0005 | 0.9515 | 190 |

One arm was rejected by the smoke gate:
- `linxyz-b075-a000`: `coord_residual_norm=0.7054`, below the smoke threshold.

### Continuation

Run:
- summary root:
  `data/runs/projres_v1c/summaries/h10016x3-qf16-v1c`
- resource:
  three continuation jobs, each `rt_QF=4` (4 nodes / 16 H100 GPUs)
- total queued allocation:
  12 nodes / 48 H100 GPUs
- epoch count:
  `CONCERTO_EPOCH=5`
- walltime requested:
  `01:10:00`

Jobs:

| job | arm | prior | beta | alpha | status | walltime | checkpoint |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| 132284.qjcm | linz-b075-a000 | linear_z | 0.75 | 0.00 | Exit 0 | 00:46:32 | `exp/concerto/arkit-full-projres-v1c-linz-b075-a000-h10016x3-qf16-v1c-continue/model/model_last.pth` |
| 132285.qjcm | mlpz-b075-a001 | mlp_z | 0.75 | 0.01 | Exit 0 | 00:46:53 | `exp/concerto/arkit-full-projres-v1c-mlpz-b075-a001-h10016x3-qf16-v1c-continue/model/model_last.pth` |
| 132286.qjcm | linxyz-b075-a001 | linear_xyz | 0.75 | 0.01 | Exit 0 | 00:47:27 | `exp/concerto/arkit-full-projres-v1c-linxyz-b075-a001-h10016x3-qf16-v1c-continue/model/model_last.pth` |

Final epoch train result:

| arm | prior | loss | enc2d | residual enc2d | alignment | target energy | removed energy | pred energy | residual norm | loss check |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linz-b075-a000 | linear_z | 7.4819 | 6.8112 | 6.4365 | 0.6603 | 0.0770 | 0.0433 | 0.0660 | 0.9169 | 0.0000 |
| mlpz-b075-a001 | mlp_z | 7.8765 | 7.6774 | 7.2917 | 1.6903 | 0.1309 | 0.0736 | 0.1690 | 0.8903 | 0.0000 |
| linxyz-b075-a001 | linear_xyz | 7.7562 | 7.3453 | 6.9669 | 1.0944 | 0.1121 | 0.0631 | 0.1094 | 0.9005 | 0.0000 |

### ARKit Stress

Source root:
`data/runs/projres_v1c/summaries/h10016x3-qf16-v1c`

Enc2d loss mean over 20 batches:

| arm | prior | clean | local surface destroy | z flip | xy swap | roll 90 x |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| linz-b075-a000 | linear_z | 7.154851 | 7.944562 | 7.561974 | 7.143595 | 7.805688 |
| mlpz-b075-a001 | mlp_z | 7.763840 | 8.748925 | 8.602632 | 7.788874 | 8.702516 |
| linxyz-b075-a001 | linear_xyz | 7.345988 | 8.572212 | 8.186496 | 7.332311 | 8.614898 |

### ScanNet Linear Gate

Source root:
`data/runs/projres_v1c/summaries/h10016x3-qf16-v1c`

| arm | prior | beta | alpha | last mIoU | best mIoU | delta last vs original | delta best vs original | delta last vs no-enc2d-renorm | delta best vs no-enc2d-renorm | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mlpz-b075-a001 | mlp_z | 0.75 | 0.01 | 0.4186 | 0.4186 | -0.0608 | -0.0366 | +0.0392 | +0.0384 | no strong-go |
| linxyz-b075-a001 | linear_xyz | 0.75 | 0.01 | 0.4110 | 0.4158 | -0.0684 | -0.0394 | +0.0316 | +0.0356 | no strong-go |
| linz-b075-a000 | linear_z | 0.75 | 0.00 | 0.4069 | 0.4069 | -0.0725 | -0.0483 | +0.0275 | +0.0267 | no strong-go |

Readout:
- Lower-capacity / height-biased priors do not beat the v1b best arm.
- `linear_z` is the most selective and flattest under stress, but downstream is
  weakest among the continued v1c arms.
- `mlp_z` is the best v1c arm, but it is still slightly below v1b
  `combo-b075-a001`.
- The next promising direction is not another static prior family sweep. The
  projection residual branch needs a more adaptive or objective-level
  intervention that removes the harmful coordinate route without globally
  subtracting useful layout/support signal.

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
