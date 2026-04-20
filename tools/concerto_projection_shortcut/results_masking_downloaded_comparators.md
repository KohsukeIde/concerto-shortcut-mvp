# Downloaded Comparator Masking Check

Attempted to extend the masking/ranking battery with downloaded external
checkpoints. The downloads succeeded, but the released PTv3 checkpoints are not
valid comparators under the current repo/data protocol.

## Downloaded Checkpoints

- PTv3 supervised ScanNet:
  `data/weights/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth`
- PTv3 PPT extreme ScanNet:
  `data/weights/ptv3/scannet-semseg-pt-v3m1-1-ppt-extreme/model/model_best.pth`
- Sonata pretraining / backbone / head:
  `data/weights/sonata/pretrain-sonata-v1m1-0-base.pth`
  `data/weights/sonata/sonata.pth`
  `data/weights/sonata/sonata_linear_prob_head_sc.pth`
- Merged Sonata linear comparator:
  `data/weights/sonata/sonata_scannet_linear_merged.pth`
- Additional downloaded heads/checkpoints for later use:
  `data/weights/concerto/concerto_large_linear_prob_head_sc.pth`
  `data/weights/utonia/utonia.pth`
  `data/weights/utonia/pretrain-utonia-v1m1-0-base_stagev2.pth`
  `data/weights/utonia/utonia_linear_prob_head_sc.pth`

## Results

| method | status | clean mIoU | random keep 20% | structured keep 20% | feature zero | clean picture | p->wall clean |
|---|---|---:|---:|---:|---:|---:|---:|
| `ptv3_supervised_scannet` | invalid protocol | 0.1496 | 0.0900 | 0.1091 | 0.0627 | 0.0937 | 0.5541 |
| `ptv3_ppt_scannet` | invalid protocol | 0.0422 | 0.0329 | 0.0411 | 0.0181 | 0.0000 | 0.4827 |
| `sonata_linear_scannet_downloaded` | valid external SSL | 0.7169 | 0.6942 | 0.6752 | 0.0607 | 0.3662 | 0.4588 |

## Interpretation

- Both PTv3 checkpoints load into the current model definitions with
  `missing=0` and `unexpected=0`, so this is not a simple key mismatch.
- The resulting clean mIoUs are far below the expected supervised/PPT ScanNet
  levels. These numbers should therefore **not** be used as supervised or PPT
  comparators in the masking/ranking table.
- The downloaded original PTv3 HF config also fails in the current code because
  it passes `cls_mode` to `PointTransformerV3`, which no longer accepts that
  argument. This supports a version/protocol mismatch interpretation.
- Sonata has a released backbone and a released ScanNet linear head. Merging
  them as `backbone.* + seg_head.*` gives a usable external SSL comparator.
  The merged model loads with `missing=0` and `unexpected=1` for the unused
  `backbone.embedding.mask_token`.
- Utonia and Concerto linear-head files were downloaded for later use. Utonia
  was not evaluated in this battery because there is no local Utonia config /
  evaluator integration in the current repo.

## Decision

Do not use the downloaded PTv3 rows as ranking evidence. Use the merged Sonata
linear row as the current external SSL comparator. To obtain a valid supervised
comparator, either train supervised PTv3 on the current `data/scannet` protocol
with the current repo, or reproduce the old released-checkpoint protocol
exactly.

Update: the released-checkpoint protocol path was reproduced with the official
Pointcept v1.5.1 model / transform code while reading the current `.npy`
ScanNet scenes. That recovers a valid supervised PTv3 row; see
`tools/concerto_projection_shortcut/results_ptv3_v151_masking_compat_full.md`.

## Files

- Summary CSV:
  `tools/concerto_projection_shortcut/results_masking_downloaded_comparators.csv`
- PTv3 supervised masking result:
  `tools/concerto_projection_shortcut/results_masking_ptv3_supervised_full.md`
- PTv3 PPT masking result:
  `tools/concerto_projection_shortcut/results_masking_ptv3_ppt_full.md`
- Sonata masking result:
  `tools/concerto_projection_shortcut/results_masking_sonata_linear_full.md`
