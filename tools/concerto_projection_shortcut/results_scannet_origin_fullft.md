# Concerto Origin ScanNet Full Fine-Tuning Result

## Run

- Experiment dir:
  - [`data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800`](/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800)
- Best checkpoint:
  - [`data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/model/model_best.pth`](/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/model/model_best.pth)
- Last checkpoint:
  - [`data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/model/model_last.pth`](/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/model/model_last.pth)
- Raw log:
  - [`data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/train.log`](/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/train.log)

## Final Validation Metrics

- mIoU: `0.8075`
- mAcc: `0.8838`
- allAcc: `0.9309`

## Class-Wise IoU Highlights

- wall: `0.8877`
- floor: `0.9593`
- cabinet: `0.7856`
- door: `0.7695`
- picture: `0.4415`
- counter: `0.7270`
- desk: `0.7569`
- shower curtain: `0.7678`
- sink: `0.7473`
- otherfurniture: `0.6892`

## Interpretation

- The released `concerto_base_origin.pth` can be full fine-tuned in the current
  repo/data environment with an official-like ScanNet recipe, and it reaches a
  strong aggregate result (`0.8075` mIoU).
- `picture` improves over the frozen linear/decoder rows, but it remains a weak
  class rather than disappearing as a bottleneck, so the downstream pathology
  is reduced but not erased by full fine-tuning.

## Confirmatory Downstream Audit

- Point-stagewise trace:
  - [`results_scannet_origin_fullft_point_stagewise_trace/scannet_point_stagewise_trace.md`](/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_scannet_origin_fullft_point_stagewise_trace/scannet_point_stagewise_trace.md)
- Oracle/actionability:
  - [`results_scannet_origin_fullft_oracle_actionability/oracle_actionability_analysis.md`](/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_scannet_origin_fullft_oracle_actionability/oracle_actionability_analysis.md)

### Stage-Wise Trace Highlights

- `picture_vs_wall`
  - point feature balanced accuracy: `0.7175`
  - refit probe on full logits: `0.7206`
  - direct pair margin: `0.7052`
- `door_vs_wall`
  - point feature balanced accuracy: `0.9548`
  - refit probe on full logits: `0.9537`
  - direct pair margin: `0.9528`
- `counter_vs_cabinet`
  - point feature balanced accuracy: `0.9510`
  - refit probe on full logits: `0.9520`
  - direct pair margin: `0.9505`

### Oracle / Actionability Highlights

- base mIoU: `0.7972`
- base picture IoU: `0.4338`
- base `picture -> wall`: `0.3956`
- `picture` top-k hit rates:
  - top-1: `0.5937`
  - top-2: `0.8525`
  - top-5: `0.9643`
- oracle upper bounds:
  - oracle top-2: mIoU `0.9165`, picture IoU `0.8304`
  - oracle top-5: mIoU `0.9744`, picture IoU `0.9567`
  - oracle graph top-5: mIoU `0.9774`, picture IoU `0.9922`

## Updated Interpretation

- Full fine-tuning improves the weak class over frozen evaluation, but it does
  not collapse the downstream audit into a trivial solved case.
- The full-FT checkpoint still has substantial actionability headroom on
  `picture_vs_wall`: top-2 already contains the ground-truth class `85.25%` of
  the time, and oracle top-5 still gives a very large upper bound.
- In other words, full FT reduces the gap but does not erase the
  readout/actionability pathology.
