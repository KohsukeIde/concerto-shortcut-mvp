# ScanNet Counterfactual Downstream Stress

## Summary

This diagnostic checks whether the released large-video ScanNet linear probe
actually depends on simple coordinate counterfactuals. It is a gate before
spending more compute on decomposed coordinate rivals.

The result is mixed but useful:

- floor-relative `z` offsets have almost no aggregate effect;
- post-center `xy` offset has almost no aggregate effect;
- compressing floor-relative height with `z_scale_050` causes a real drop;
- the biggest class damage under `z_scale_050` is `picture`, `counter`,
  `sink`, and `door`.

This means ScanNet20 linear downstream is not strongly sensitive to absolute
coordinate offsets, but it does use relative vertical scale / height structure.

## Setup

- Job: `133146.qjcm`
- Resource: `rt_QF=1`, walltime `00:40:00`
- Checkpoint:
  `exp/concerto/scannet-proxy-large-video-official-lin/model/model_last.pth`
- Config:
  `configs/concerto/semseg-ptv3-large-v1m1-0a-scannet-lin-proxy-valonly.py`
- Full val batches: `312`
- Outputs:
  `data/runs/scannet_counterfactual_downstream/large_video_official/counterfactual_summary.md`
  and `counterfactual_stress_classwise.csv`

## Overall

| stress | mIoU | delta vs clean | mAcc | allAcc |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.7681 | +0.0000 | 0.8649 | 0.9158 |
| z_shift_p025 | 0.7681 | -0.0001 | 0.8647 | 0.9157 |
| z_shift_p050 | 0.7679 | -0.0002 | 0.8645 | 0.9156 |
| z_shift_p100 | 0.7689 | +0.0008 | 0.8648 | 0.9158 |
| z_scale_050 | 0.7561 | -0.0120 | 0.8535 | 0.9116 |
| z_scale_150 | 0.7669 | -0.0012 | 0.8621 | 0.9145 |
| xy_shift_post_p050 | 0.7679 | -0.0003 | 0.8651 | 0.9157 |

## Focus Classes

The key class deltas against clean are:

| stress | notable class deltas |
| --- | --- |
| z_shift_p025 | picture `-0.0039`, cabinet `-0.0031`, sink `+0.0026` |
| z_shift_p050 | cabinet `-0.0034`, door `-0.0029`, picture `-0.0028` |
| z_shift_p100 | cabinet `-0.0060`, picture `+0.0057`, sink `+0.0047` |
| z_scale_050 | picture `-0.0388`, counter `-0.0370`, sink `-0.0185`, door `-0.0154` |
| z_scale_150 | cabinet `-0.0128`, counter `+0.0131`, sink `+0.0090`, shower curtain `-0.0085` |
| xy_shift_post_p050 | picture `+0.0081`, desk `+0.0038`, cabinet `-0.0022` |

## Interpretation

The cheap counterfactual gate does not support a strong claim that ScanNet20
linear probing depends on absolute coordinate offsets. The model is almost
invariant to constant floor-relative z shifts and to a post-center xy shift.

However, the model is sensitive to vertical scale. Compressing relative height
by 0.5 drops mIoU by `0.0120`, with the biggest losses on `picture`,
`counter`, `sink`, and `door`. This is closer to a scale/shape dependence than
to an absolute-position lookup.

This narrows the next step:

- Do not jump to a broad decomposed rival for absolute `x/y/z` offsets.
- If continuing the coordinate line, focus on relative-height scale and
  class/confusion-aware effects.
- The strongest downstream bottleneck remains `picture -> wall` and furniture
  cluster confusion, so a class-aware intervention is still more justified
  than another global coordinate-margin sweep.
