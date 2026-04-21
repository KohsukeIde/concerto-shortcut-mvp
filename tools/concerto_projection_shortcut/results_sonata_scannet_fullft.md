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
- A pairwise stagewise/oracle audit on this full-FT checkpoint is still a
  separate follow-up if we want exact parity with the existing Concerto /
  Sonata / PTv3 downstream-audit tables.
