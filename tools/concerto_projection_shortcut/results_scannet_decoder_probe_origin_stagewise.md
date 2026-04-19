# ScanNet Origin Decoder Stage-Wise Trace

## Summary

This run repeats the point-level stage-wise trace on the main/origin line,
using the `concerto_base_origin.pth` decoder-probe checkpoint instead of any
large-video checkpoint.

Key result:

- `picture_vs_wall` is separable in the frozen origin decoder point feature:
  `balanced acc = 0.8376`, `AUC = 0.9464`.
- A refit binary probe on the decoder 20-way logits remains similar:
  `balanced acc = 0.8247`, `AUC = 0.9363`.
- But the fixed 20-way direct margin
  `logit(picture) - logit(wall)` is much weaker:
  `balanced acc = 0.7203`, despite `AUC = 0.9292`.
- On the sampled `picture_vs_wall` subset, `54.96%` of target `picture`
  points are predicted as `wall`.

This supports a more specific read than "decoder capacity fixes it":
the origin decoder features contain meaningful `picture/wall` information, but
the final 20-way readout still underuses it for `picture`.

## Setup

- Job: `133330.qjcm`.
- Resource: `rt_QF=1`, one visible GPU through `GPU_IDS_CSV=0`.
- Walltime used: `00:02:26`.
- Config:
  `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/config.py`.
- Weight:
  `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`.
- Output directory:
  `data/runs/scannet_decoder_probe_origin/stagewise_trace_best`.
- Train batches: `256`.
- Val batches: `128`.
- Max samples per class: `60000`.
- Bootstrap iters: `100`.

Outputs:

- `data/runs/scannet_decoder_probe_origin/stagewise_trace_best/scannet_point_stagewise_trace.md`
- `data/runs/scannet_decoder_probe_origin/stagewise_trace_best/scannet_point_stagewise_trace.csv`
- `data/runs/scannet_decoder_probe_origin/stagewise_trace_best/scannet_point_stagewise_trace_confusion.csv`
- `data/runs/scannet_decoder_probe_origin/stagewise_trace_best/metadata.json`
- `data/logs/abciq/133330.qjcm.OU`

## Main Results

| pair | point feature bal acc | 20-way logits refit bal acc | direct class-margin bal acc |
| --- | ---: | ---: | ---: |
| picture_vs_wall | 0.8376 | 0.8247 | 0.7203 |
| counter_vs_cabinet | 0.9498 | 0.9484 | 0.9389 |
| desk_vs_wall | 0.9847 | 0.9808 | 0.9734 |
| desk_vs_table | 0.9058 | 0.9060 | 0.9098 |
| sink_vs_cabinet | 0.9644 | 0.9678 | 0.9550 |
| door_vs_wall | 0.9622 | 0.9633 | 0.9585 |
| shower_curtain_vs_wall | 0.9741 | 0.9761 | 0.9468 |

## Picture/Wall Confusion

On the sampled `picture_vs_wall` validation subset:

| target | predicted | fraction |
| --- | --- | ---: |
| picture | wall | 0.5496 |
| picture | picture | 0.4254 |
| wall | wall | 0.9758 |
| wall | picture | 0.0015 |

## Interpretation

- This origin-only trace should replace large-video-based point-stage traces
  for mainline decisions.
- The class-wise problem is not that the frozen origin decoder feature cannot
  separate `picture` from `wall`; a binary probe can recover that separation.
- The problem is sharper: the trained 20-way decoder readout strongly favors
  `wall` over `picture` on actual target `picture` points.
- This makes a confusion-aware or class-calibrated readout/intervention more
  plausible than another global coordinate-rival sweep.
