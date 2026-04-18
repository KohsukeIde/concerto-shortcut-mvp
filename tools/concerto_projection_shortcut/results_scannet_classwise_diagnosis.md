# ScanNet Class-wise Diagnosis

## Summary

This run analyzes why aggregate ScanNet linear mIoU does not improve under
the current SR-LoRA v5 line. It compares the same-starting-checkpoint large
video official linear probe against two SR-LoRA follow-up checkpoints:

- baseline: `scannet-proxy-large-video-official-lin`
- SR-LoRA m=0.1: `scannet-proxy-sr-lora-v5-r4-d0p3-i256-qf4-lin`
- SR-LoRA m=0.2: `scannet-proxy-sr-lora-v5-r4-d0p3-m0p2-i256-qf4-lin`

The main conclusion is that the downstream bottleneck is class/confusion
specific. A global coordinate-rival margin does not target the weakest
failure modes strongly enough.

## Outputs

- Log-derived classwise table:
  `data/runs/scannet_classwise_diagnosis/large_video_sr_lora/classwise_from_logs.md`
- Full clean-val confusion summary:
  `data/runs/scannet_classwise_diagnosis/large_video_sr_lora/classwise_confusion_summary.md`
- Per-model class metrics and confusion CSVs:
  `data/runs/scannet_classwise_diagnosis/large_video_sr_lora/*_class_metrics.csv`
  and `*_top_confusions.csv`
- ABCI-Q job:
  `133141.qjcm`, `rt_QF=1`, walltime `00:50:00`, completed.

## Overall

The clean-val confusion evaluation used all 312 ScanNet val batches.

| experiment | mIoU | mAcc | allAcc |
| --- | ---: | ---: | ---: |
| baseline | 0.7681 | 0.8649 | 0.9158 |
| SR-LoRA m=0.1 | 0.7688 | 0.8655 | 0.9156 |
| SR-LoRA m=0.2 | 0.7683 | 0.8637 | 0.9153 |

These match the earlier follow-up readout: the SR-LoRA line is essentially
neutral in aggregate mIoU and does not pass the downstream gate.

## Weak Classes

Baseline class IoU is highly uneven.

| rank | class | IoU | accuracy | main non-self confusion |
| ---: | --- | ---: | ---: | --- |
| 1 | picture | 0.3962 | 0.5347 | wall, 44.3% of target picture |
| 2 | counter | 0.6543 | 0.8394 | cabinet, 9.3% |
| 3 | desk | 0.6790 | 0.8396 | wall/table/bookshelf, each about 3-4% |
| 4 | sink | 0.6859 | 0.8290 | cabinet 7.6%, counter 7.3% |
| 5 | otherfurniture | 0.6929 | 0.7807 | wall 6.8% |
| 6 | cabinet | 0.6992 | 0.8099 | wall 5.6%, otherfurniture 4.3% |
| 7 | shower curtain | 0.7179 | 0.7704 | wall 16.6%, bathtub 3.4% |
| 8 | door | 0.7579 | 0.8928 | wall 8.4% |

The strongest failure is not a generic height or coordinate issue. It is a
semantic/appearance boundary issue where thin or wall-mounted classes are
absorbed into `wall`, plus a furniture confusion cluster around
`counter/cabinet/sink` and `desk/table/bookshelf`.

## SR-LoRA Class Deltas

SR-LoRA m=0.1:

| largest gains | delta IoU |
| --- | ---: |
| shower curtain | +0.0074 |
| sink | +0.0069 |
| picture | +0.0022 |

| largest drops | delta IoU |
| --- | ---: |
| cabinet | -0.0034 |
| refridgerator | -0.0030 |
| table | -0.0011 |

SR-LoRA m=0.2:

| largest gains | delta IoU |
| --- | ---: |
| shower curtain | +0.0083 |
| sink | +0.0074 |
| sofa | +0.0041 |

| largest drops | delta IoU |
| --- | ---: |
| cabinet | -0.0054 |
| window | -0.0049 |
| picture | -0.0024 |

The positive movement is real but too narrow and partly offset by drops in
other classes. Increasing the margin from 0.1 to 0.2 does not solve the core
weak class `picture`; it worsens it.

## Interpretation

The current SR-LoRA v5 similarity-margin objective is a stable training
intervention, but it is not aligned with the actual ScanNet bottleneck.
It weakly helps `sink` and `shower curtain`, while the most severe baseline
failure, `picture -> wall`, remains unresolved or worsens under the stronger
margin.

Therefore, the next intervention should not be another broad coordinate-rival
sweep. It should be class-aware or confusion-aware:

- target `picture -> wall` explicitly;
- target `shower curtain -> wall/bathtub` if keeping the one class where
  SR-LoRA already shows a consistent positive signal;
- target `sink/counter/cabinet` and `desk/table/bookshelf` as furniture
  confusion clusters;
- use the 3D anchor and enc2d surgery as tools, but condition the pressure on
  classes or pseudo-confusion rather than applying a global coordinate margin.

This also reframes the prior negative result: shortcut suppression alone is
not a sufficient downstream objective. The missing step is to redirect the
freed capacity toward specific semantic confusions.
