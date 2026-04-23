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
- The next confirmatory step, if needed for the paper, is to run the same
  pairwise stagewise/oracle audit on this full-FT checkpoint.
