# ScanNet Decoder Probe On Concerto Origin

## Summary

This run tests whether a stronger frozen-encoder decoder probe fixes the
class-wise bottleneck observed in ScanNet linear probing, especially
`picture -> wall`.

The answer is currently no. A 100 epoch decoder probe reaches strong aggregate
ScanNet validation performance, but `picture` remains low:

- Final precise eval: `mIoU/mAcc/allAcc = 0.7888 / 0.8813 / 0.9243`.
- `picture IoU = 0.4217`, still far below the easy classes and not a
  decoder-probe resolution of the `picture -> wall` failure.
- The job completed successfully in `00:34:04` on `rt_QF=2`.

This does not replace a full-FT per-class check, but it rules out the cheapest
"just add the Concerto decoder probe" explanation for the weak `picture`
class.

## Setup

- Weight: `data/weights/concerto/concerto_base_origin.pth`.
- Model/config family: `semseg-ptv3-base-v1m1-0c-scannet-dec`.
- Added configs:
  - `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e001.py`
  - `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Checkpoint loading follows the resolved Pointcept issue guidance for
  released HF/origin weights:
  - `CheckpointLoader(keywords="module", replacement="module.backbone")`
- ABCI-Q wrapper:
  - `tools/concerto_projection_shortcut/submit_scannet_semseg_train_abciq_qf.sh`
- Run root:
  - `data/runs/scannet_decoder_probe_origin`

## Jobs

| experiment | job | resource | walltime used | status |
| --- | --- | --- | ---: | --- |
| `scannet-dec-origin-e001` | `133216.qjcm` | `rt_QF=2` | `00:10:19` | `Exit_status=0` |
| `scannet-dec-origin-e100` | `133217.qjcm` | `rt_QF=2` | `00:34:04` | `Exit_status=0` |

Logs:

- `data/logs/abciq/scannet_semseg_133216.qjcm.log`
- `data/logs/abciq/scannet_semseg_133217.qjcm.log`
- `data/runs/scannet_decoder_probe_origin/logs/multinode/133216.qjcm_scannet-dec-origin-e001_20260419_031517/logs/qh140.rank0.scannet-dec-origin-e001.log`
- `data/runs/scannet_decoder_probe_origin/logs/multinode/133217.qjcm_scannet-dec-origin-e100_20260419_032633/logs/qh140.rank0.scannet-dec-origin-e100.log`

Checkpoints:

- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e001/model/model_last.pth`
- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_last.pth`
- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`

Class-wise outputs computed from the saved validation predictions:

- `data/runs/scannet_decoder_probe_origin/classwise/scannet_dec_origin_e100_summary.json`
- `data/runs/scannet_decoder_probe_origin/classwise/scannet_dec_origin_e100_class_metrics.csv`
- `data/runs/scannet_decoder_probe_origin/classwise/scannet_dec_origin_e100_top_confusions.csv`
- `data/runs/scannet_decoder_probe_origin/classwise/scannet_dec_origin_e100_confusion_long.csv`
- `data/runs/scannet_decoder_probe_origin/stagewise_trace_best/scannet_point_stagewise_trace.md`
- `data/runs/scannet_decoder_probe_origin/stagewise_trace_best/scannet_point_stagewise_trace.csv`
- `data/runs/scannet_decoder_probe_origin/stagewise_trace_best/scannet_point_stagewise_trace_confusion.csv`

## Final 100 Epoch Result

Final precise eval:

| metric | value |
| --- | ---: |
| mIoU | 0.7888 |
| mAcc | 0.8813 |
| allAcc | 0.9243 |

Per-class final IoU:

| class | IoU | accuracy |
| --- | ---: | ---: |
| wall | 0.8793 | 0.9356 |
| floor | 0.9584 | 0.9794 |
| cabinet | 0.7318 | 0.8337 |
| bed | 0.8432 | 0.8875 |
| chair | 0.9183 | 0.9554 |
| sofa | 0.7931 | 0.9358 |
| table | 0.7896 | 0.8491 |
| door | 0.7715 | 0.9129 |
| window | 0.7630 | 0.8637 |
| bookshelf | 0.8262 | 0.9122 |
| picture | 0.4217 | 0.5472 |
| counter | 0.7044 | 0.8209 |
| desk | 0.7096 | 0.8676 |
| curtain | 0.8634 | 0.9192 |
| refridgerator | 0.7647 | 0.9031 |
| shower curtain | 0.8055 | 0.8927 |
| toilet | 0.9531 | 0.9780 |
| sink | 0.7199 | 0.8594 |
| bathtub | 0.8481 | 0.9662 |
| otherfurniture | 0.7117 | 0.8069 |

Weakest classes after decoder probe:

| rank | class | IoU | accuracy | main non-self confusion |
| ---: | --- | ---: | ---: | --- |
| 1 | picture | 0.4217 | 0.5472 | wall, 43.1% of target picture |
| 2 | counter | 0.7044 | 0.8209 | cabinet, 9.0% |
| 3 | desk | 0.7096 | 0.8676 | table, 6.1% |
| 4 | otherfurniture | 0.7117 | 0.8069 | wall, 6.4% |
| 5 | sink | 0.7199 | 0.8594 | counter, 5.0% |
| 6 | cabinet | 0.7318 | 0.8337 | otherfurniture, 4.8% |
| 7 | window | 0.7630 | 0.8637 | wall, 9.0% |
| 8 | refridgerator | 0.7647 | 0.9031 | cabinet, 3.3% |

Top confusions for `picture`:

| target | predicted | fraction of target |
| --- | --- | ---: |
| picture | wall | 0.4310 |
| picture | sofa | 0.0068 |
| picture | bookshelf | 0.0037 |
| picture | otherfurniture | 0.0031 |
| picture | bed | 0.0022 |

## Intermediate Readout

The decoder probe already reached strong aggregate mIoU early, but `picture`
did not move into a high-IoU regime:

| checkpoint in log | mIoU | picture IoU | picture acc |
| --- | ---: | ---: | ---: |
| epoch 16 eval | 0.7694 | 0.4054 | 0.5435 |
| epoch 41 eval | 0.7682 | 0.3964 | 0.5826 |
| epoch 82 eval | 0.7747 | 0.3903 | 0.4935 |
| final precise eval | 0.7888 | 0.4217 | 0.5472 |

## Interpretation

- Decoder capacity improves aggregate validation performance relative to the
  previous linear-probe diagnostic regime, as expected.
- The main weak class from the class-wise diagnosis remains weak: `picture`
  is only `0.4217` IoU after the 100 epoch decoder probe, and `43.1%` of
  target `picture` points are still predicted as `wall`.
- Therefore, the `picture -> wall` issue is not explained away by the cheapest
  "linear probe is too weak; a frozen-encoder decoder probe fixes it" scenario.
- A full-FT per-class check is still a separate question. The decoder result
  does not prove that full FT cannot improve `picture`, but it makes the
  decoder/readout-only explanation much weaker.

## Origin Decoder Stage-Wise Trace

The origin decoder point-stage trace was run separately with no large-video
checkpoint involved:

- Job: `133330.qjcm`, `rt_QF=1`, walltime used `00:02:26`.
- Config:
  `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/config.py`.
- Weight:
  `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`.
- Result file:
  `tools/concerto_projection_shortcut/results_scannet_decoder_probe_origin_stagewise.md`.

For `picture_vs_wall`, a binary probe on the decoder point feature reaches
`0.8376` balanced accuracy, and a binary probe on the 20-way logits reaches
`0.8247`. However, the fixed direct class margin
`logit(picture) - logit(wall)` reaches only `0.7203`, and `54.96%` of sampled
target `picture` points are predicted as `wall`.

This means the origin decoder features do contain useful `picture/wall`
information, but the trained 20-way readout still underuses it for `picture`.
