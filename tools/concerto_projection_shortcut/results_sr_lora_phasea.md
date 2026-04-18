# SR-LoRA v5 Phase A Results

## Summary

SR-LoRA v5 Phase A was implemented and tested on the released large-video
full checkpoint with a six-indoor-dataset training mix and the frozen
coord-only rival from the main-variant Step 0.5 diagnostic.

The line is stable at the training level, but the downstream gate is no-go:

- `m=0.1`, `rank=4`, `distill=0.3` gives only a tiny clean ScanNet linear
  increase and does not improve stress robustness.
- Increasing the margin to `m=0.2` increases the training-side margin pressure
  but does not improve downstream metrics.
- No tested SR-LoRA setting reaches the gate of clean `+0.005` or stress
  `+0.005`.
- The remaining three `m=0.1` matrix arms were not downstream-evaluated because
  the selected representative arm did not meet the gate.

## Fixed Setup

- Starting checkpoint:
  `data/weights/concerto/pretrain-concerto-v1m1-2-large-video.pth`.
- Rival checkpoint:
  `data/runs/main_variant_coord_mlp_rival/main-origin-six-step05/model_last.pth`.
- Phase A config:
  `configs/concerto/pretrain-concerto-v1m1-2-large-video-sr-lora-v5-phasea.py`.
- LoRA injection:
  `student.backbone.enc.*.attn.qkv`.
- Trainable parameters:
  LoRA parameters only, `156,672` params for rank 4.
- Training data:
  six indoor Concerto datasets, excluding RE10K.
- Linear/stress comparison baseline:
  same starting checkpoint, `pretrain-concerto-v1m1-2-large-video.pth`.
- ScanNet linear config:
  `configs/concerto/semseg-ptv3-large-v1m1-0a-scannet-lin-proxy-valonly.py`.

## Training Runs

| stage | job | exp | rank | distill | margin | type | enc2d | margin loss | full sim | rival sim | distill loss | delta norm | status |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | `133093.qjcm` | `sr-lora-v5-smoke3-r4-d03` | 4 | 0.3 | 0.1 | similarity | 5.2760 | 0.0139 | 0.4223 | 0.2595 | 0.0288 | - | pass |
| pilot | `133095.qjcm` | `sr-lora-v5-pilot-r4-d03-i256-qf4` | 4 | 0.3 | 0.1 | similarity | 4.0167 | 0.0148 | 0.6026 | 0.3633 | 0.0997 | 2.7964 | pass |
| matrix | `133097.qjcm` | `sr-lora-v5-r4-d0p3-i256-qf4-matrix` | 4 | 0.3 | 0.1 | similarity | 4.0095 | 0.0147 | 0.6035 | 0.3635 | 0.0981 | 2.5362 | pass |
| matrix | `133098.qjcm` | `sr-lora-v5-r4-d1p0-i256-qf4-matrix` | 4 | 1.0 | 0.1 | similarity | 4.0698 | 0.0147 | 0.6027 | 0.3630 | 0.0878 | 1.9275 | pass |
| matrix | `133099.qjcm` | `sr-lora-v5-r8-d0p3-i256-qf4-matrix` | 8 | 0.3 | 0.1 | similarity | 4.0130 | 0.0147 | 0.6030 | 0.3632 | 0.0989 | 3.7434 | pass |
| matrix | `133100.qjcm` | `sr-lora-v5-r8-d1p0-i256-qf4-matrix` | 8 | 1.0 | 0.1 | similarity | 4.0665 | 0.0147 | 0.6031 | 0.3633 | 0.0862 | 3.2764 | pass |
| pilot | `133104.qjcm` | `sr-lora-v5-r4-d0p3-m0p2-i256-qf4` | 4 | 0.3 | 0.2 | similarity | 4.0404 | 0.0443 | 0.6029 | 0.3630 | 0.0980 | 2.6529 | pass |
| smoke | `133137.qjcm` | `sr-lora-v5-lossmargin-smoke-r4-d03-lm1p0` | 4 | 0.3 | 1.0 | loss | 3.8486 | 0.1153 | 0.6300 | 0.3690 | 0.1107 | 0.0023 | pass |

Notes:

- The 4-condition matrix showed very similar SR metrics across rank and
  distill settings. This was interpreted as early saturation of the current
  metric rather than an unstable or failed training run.
- The representative `r4 d0.3` arm was therefore selected for downstream
  follow-up before spending more evaluation time on the rest of the matrix.
- The `m=0.2` pilot increased `sr_margin_loss` from roughly `0.015` to roughly
  `0.044`, confirming stronger margin pressure without immediate training
  collapse.
- A 2-iteration loss-based hinge smoke completed after the no-go follow-ups.
  It used `SR_MARGIN_TYPE=loss` and `SR_MARGIN_VALUE=1.0`, where the margin is
  in the same `*10` enc2d-loss units as `enc2d_alignment_loss`. The run logged
  `sr_full_loss=3.7000`, `sr_rival_loss=6.3098`, and `sr_margin_loss=0.1153`,
  confirming that the loss-margin branch is wired and active.

## Downstream Follow-up

| margin | job | SR linear exp | last mIoU | best mIoU | linear delta last | linear delta best | decision |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0.1 | `133102.qjcm` | `scannet-proxy-sr-lora-v5-r4-d0p3-i256-qf4-lin` | 0.7694 | 0.7700 | +0.0009 | +0.0015 | no-go |
| 0.2 | `133105.qjcm` | `scannet-proxy-sr-lora-v5-r4-d0p3-m0p2-i256-qf4-lin` | 0.7682 | 0.7682 | -0.0003 | -0.0003 | no-go |

Baseline:

- Linear exp: `scannet-proxy-large-video-official-lin`.
- Last/best mIoU: `0.7685 / 0.7685`.

## Stress Follow-up

| margin | clean | local surface destroy | z flip | xy swap | roll 90 x |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | +0.0007 | -0.0011 | -0.0044 | +0.0010 | +0.0002 |
| 0.2 | +0.0002 | -0.0005 | -0.0030 | +0.0001 | +0.0004 |

These are SR-LoRA minus same-starting-checkpoint baseline deltas on ScanNet
validation, using 312 batches per stress condition.

## Failed / Corrected Run

- `133101.qjcm` was the first downstream follow-up attempt.
- It failed because the base ScanNet linear config was accidentally used with
  the large-video checkpoint, causing a shape mismatch.
- The failed directories were moved aside with a `failed-base-config` suffix.
- The corrected large-config follow-up is `133102.qjcm`.

## Output Locations

- CSV summary:
  `tools/concerto_projection_shortcut/results_sr_lora_phasea.csv`.
- `m=0.1` follow-up:
  `data/runs/sr_lora_phasea/followup/r4-d0p3-i256/followup_compare.md`.
- `m=0.2` follow-up:
  `data/runs/sr_lora_phasea/followup/r4-d0p3-m0p2-i256/followup_compare.md`.
- `m=0.1` merged checkpoint:
  `exp/concerto/sr-lora-v5-r4-d0p3-i256-qf4-matrix/model/model_last_merged_lora.pth`.
- `m=0.2` merged checkpoint:
  `exp/concerto/sr-lora-v5-r4-d0p3-m0p2-i256-qf4/model/model_last_merged_lora.pth`.

## Interpretation

The coord-only frozen rival is useful as a diagnostic lower bound, but the
current SR-LoRA v5 objective does not create a downstream-relevant change in the
large-video checkpoint representation. The most direct pressure increase
tested so far, margin `m=0.2`, did not improve ScanNet linear or stress
metrics.

Do not launch the remaining similarity-margin matrix follow-ups, a longer
similarity-margin run, or a larger similarity-margin matrix without a new
hypothesis. The loss-based hinge branch is now available as a separate minimal
ablation, but it only has a 2-iteration smoke so far and has not yet passed a
downstream gate.
