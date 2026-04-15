# ProjRes v1 Results

Updated: 2026-04-16 01:05 JST

## Bottom Line

`projres_v1a` completed the H100 continuation, ARKit stress, and ScanNet
linear gate. The ScanNet gate is a no-go for this arm.

The fix did not match the original Concerto continuation on the ScanNet linear
proxy. It also landed slightly below the `no-enc2d-renorm` control.

## Run

| item | value |
| --- | --- |
| selected alpha | `0.05` |
| selected prior | `mlp` |
| continuation experiment | `arkit-full-projres-v1a-alpha005-h10032-qf32-continue` |
| continuation job | `132196.qjcm`, `Exit_status=0`, walltime `00:39:37` |
| follow-up job | `132198.qjcm`, `Exit_status=0`, walltime `00:50:06` |
| continuation checkpoint | `exp/concerto/arkit-full-projres-v1a-alpha005-h10032-qf32-continue/model/model_last.pth` |
| linear experiment | `scannet-proxy-projres-v1a-alpha005-h10032-qf32-lin` |

## Continuation

Final epoch train result:

| loss | enc2d loss | residual enc2d loss | alignment | target energy | pred energy | residual norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8.0899 | 8.0655 | 6.3309 | 0.0200 | 0.1061 | 0.0020 | 0.7308 |

## ARKit Stress

Source:
`data/runs/projres_v1/summaries/h10032-qf32/arkit-full-projres-v1a-alpha005-h10032-qf32-continue_stress.csv`

| stress | batches | enc2d loss mean |
| --- | ---: | ---: |
| clean | 20 | 8.022868 |
| local_surface_destroy | 20 | 9.115467 |
| z_flip | 20 | 8.941781 |
| xy_swap | 20 | 8.022733 |
| roll_90_x | 20 | 9.353544 |

## ScanNet Linear Gate

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
