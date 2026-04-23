# Sonata ScanNet Full Fine-Tuning Result

## Run

- Experiment dir:
  - [`data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800`](/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800)
- Best checkpoint:
  - [`data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/model/model_best.pth`](/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/model/model_best.pth)
- Last checkpoint:
  - [`data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/model/model_last.pth`](/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/model/model_last.pth)
- Raw log:
  - [`data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/train.log`](/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800/train.log)

## Final Validation Metrics

- mIoU: `0.7955`
- mAcc: `0.8649`
- allAcc: `0.9271`

## Class-Wise IoU Highlights

- wall: `0.8867`
- floor: `0.9512`
- cabinet: `0.7704`
- door: `0.7635`
- picture: `0.3602`
- counter: `0.7284`
- desk: `0.8020`
- shower curtain: `0.7482`
- sink: `0.7337`
- otherfurniture: `0.6674`

## Interpretation

- This confirms that the official-like Sonata ScanNet full-FT line is runnable
  in the current repo/data environment and yields a strong external SSL anchor
  beyond the released frozen linear-head comparator.
- `picture` remains weak (`0.3602`) even though aggregate mIoU is strong
  (`0.7955`), so the downstream weak-class issue does not disappear merely by
  moving from frozen linear evaluation to full fine-tuning.

## Confirmatory Downstream Audit

- Point-stagewise trace:
  - [`results_sonata_fullft_point_stagewise_trace/scannet_point_stagewise_trace.md`](/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_sonata_fullft_point_stagewise_trace/scannet_point_stagewise_trace.md)
- Oracle/actionability:
  - [`results_sonata_fullft_oracle_actionability/oracle_actionability_analysis.md`](/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_sonata_fullft_oracle_actionability/oracle_actionability_analysis.md)

### Stage-Wise Trace Highlights

- `picture_vs_wall`
  - point feature balanced accuracy: `0.6742`
  - refit probe on full logits: `0.6734`
  - direct pair margin: `0.6485`
- `door_vs_wall`
  - point feature balanced accuracy: `0.9550`
  - refit probe on full logits: `0.9598`
  - direct pair margin: `0.9445`
- `counter_vs_cabinet`
  - point feature balanced accuracy: `0.9576`
  - refit probe on full logits: `0.9596`
  - direct pair margin: `0.9469`

### Oracle / Actionability Highlights

- base mIoU: `0.7770`
- base picture IoU: `0.3508`
- base `picture -> wall`: `0.5478`
- `picture` top-k hit rates:
  - top-1: `0.4366`
  - top-2: `0.6206`
  - top-5: `0.7746`
- oracle upper bounds:
  - oracle top-2: mIoU `0.8856`, picture IoU `0.6003`
  - oracle top-5: mIoU `0.9519`, picture IoU `0.7700`
  - oracle graph top-5: mIoU `0.9663`, picture IoU `0.9924`

## Updated Interpretation

- Full fine-tuning does not make the Sonata row behave like a cleanly solved
  supervised readout.
- `picture` remains weak, `picture -> wall` remains high, and the candidate-set
  oracle headroom is still large.
- This keeps Sonata useful as a backbone-moving external SSL anchor for the ED
  framing, rather than collapsing it into a trivial no-gap comparator.
