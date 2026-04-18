# DINO Patch Bias Step A'

## Setup
- model: `facebook/dinov2-with-registers-giant`
- data root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/concerto_scannet_imagepoint_absmeta`
- train images seen: 512
- val images seen: 256
- train semantic samples: 3667
- val semantic samples: 1878
- train position samples: 24000
- val position samples: 16384

## Results

| variant | pic/wall bal acc | pic acc | wall acc | AUC | pos R2 mean | pos R2 row | pos R2 col |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_dino | 0.7797 | 0.5920 | 0.9675 | 0.8827 | 0.8787 | 0.8917 | 0.8657 |
| rasa_lite_position_removed | 0.7837 | 0.6000 | 0.9675 | 0.8843 | 0.8651 | 0.8909 | 0.8394 |

## Interpretation Guide
- High picture/wall accuracy means DINO patch features carry separable 2D semantics for this failure pair.
- High position R2 means DINO patch features linearly expose patch location.
- If RASA-lite lowers position R2 while retaining picture/wall accuracy, teacher-side target debiasing is plausible.
