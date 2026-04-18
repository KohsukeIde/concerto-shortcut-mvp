# Concerto 3D Patch Separation Step A

## Superseded Control Note

This first-pass A'' comparison is superseded by
`results_concerto3d_dino_exact_controls_stepA.md` for the main interpretation.
The exact-patch control shows that the earlier DINO Step A' `0.7797` vs
Concerto `0.5381/0.5547` gap was not apples-to-apples: on the exact same A''
patch subset, `picture_vs_wall` DINO balanced accuracy is `0.5801`, close to
Concerto `encoder_pooled` `0.5772`.

## Setup
- config: `pretrain-concerto-v1m1-2-large-video`
- weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/concerto/pretrain-concerto-v1m1-2-large-video.pth`
- data root: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/concerto_scannet_imagepoint_absmeta`
- job: `133152.qjcm`, `Exit_status=0`, walltime `00:04:51`
- train batches seen: 256
- val batches seen: 128
- train class counts: {'wall': 12000, 'picture': 245}
- val class counts: {'wall': 9219, 'picture': 78}
- note: `DefaultImagePointDataset` reads `segment.npy`, while this ScanNet
  image-point root stores labels as `segment20.npy`; the runner creates
  lightweight `segment.npy -> segment20.npy` aliases before extraction.

## Results

| variant | pic/wall bal acc | pic acc | wall acc | AUC | pos R2 mean |
|---|---:|---:|---:|---:|---:|
| encoder_pooled | 0.5381 | 0.0897 | 0.9865 | 0.6662 | -1.4003 |
| patch_proj | 0.5547 | 0.1154 | 0.9939 | 0.6980 | -1.3971 |

## DINO Step A' Comparison

| source | feature | pic/wall bal acc | AUC |
|---|---|---:|---:|
| DINOv2 | raw patch feature | 0.7797 | 0.8827 |
| Concerto 3D | encoder pooled to patches | 0.5381 | 0.6662 |
| Concerto 3D | after enc2d patch projection | 0.5547 | 0.6980 |

## Interpretation Guide
- `encoder_pooled` is the Concerto 3D encoder feature pooled to image patches through point-pixel correspondences.
- `patch_proj` is the feature after Concerto's enc2d patch projection head.
- Compare picture/wall balanced accuracy to DINO Step A' to see whether 3D alignment preserves or loses this 2D semantic separation.
- On this ScanNet `picture`/`wall` diagnostic, DINOv2 patch features are much more linearly separable than Concerto's corresponding 3D patch-pooled features.
- This supports a semantic transfer / 3D alignment bottleneck for the `picture -> wall` failure pair, rather than a DINO-only failure.
- The position R2 column here is computed on the semantic `picture`/`wall` patch subset, not on the all-patch position set used by `results_dino_patch_bias_stepA.md`; use it only as a secondary diagnostic.
