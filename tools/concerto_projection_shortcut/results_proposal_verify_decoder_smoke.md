# Proposal-then-Verify Decoder Pilot

Lightweight PVD pilot: learn a hard-class proposal verifier on fine voxel proposals and fuse conservative proposal boosts into base decoder logits.

## Setup

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Region voxel size: `4`
- Positive purity: `0.8`
- Proposal classes: `picture,counter,desk,sink,door,shower curtain,cabinet,table,wall`
- Train batches: `8`
- Train proposal count: `51041`
- Seen val batches: `8`

## Headline

- Base: mIoU=0.5641, picture=0.6800, picture->wall=0.0254
- Best mIoU: `base` mIoU=0.5641 (Δ+0.0000), picture=0.6800 (Δ+0.0000), p->wall=0.0254 (Δ+0.0000)
- Best safe picture: `pvd_thr0p5_b0p25` mIoU=0.5634 (Δ-0.0007), picture=0.6851 (Δ+0.0051), p->wall=0.0248 (Δ-0.0006)

## Top Variants

| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | p->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `pvd_thr0p9_b0p25` | 0.5639 | -0.0002 | 0.5383 | +0.0001 | 0.6804 | +0.0004 | 0.0260 | +0.0006 |
| 2 | `pvd_thr0p7_b0p25` | 0.5639 | -0.0002 | 0.5375 | -0.0007 | 0.6826 | +0.0025 | 0.0254 | +0.0000 |
| 3 | `pvd_thr0p9_b0p5` | 0.5636 | -0.0004 | 0.5380 | -0.0002 | 0.6804 | +0.0004 | 0.0260 | +0.0006 |
| 4 | `pvd_thr0p9_b1p0` | 0.5635 | -0.0006 | 0.5381 | -0.0002 | 0.6804 | +0.0004 | 0.0260 | +0.0006 |
| 5 | `pvd_thr0p5_b0p25` | 0.5634 | -0.0007 | 0.5358 | -0.0025 | 0.6851 | +0.0051 | 0.0248 | -0.0006 |
| 6 | `pvd_thr0p7_b0p5` | 0.5627 | -0.0014 | 0.5365 | -0.0017 | 0.6843 | +0.0042 | 0.0254 | +0.0000 |
| 7 | `pvd_thr0p7_b1p0` | 0.5619 | -0.0022 | 0.5357 | -0.0025 | 0.6820 | +0.0020 | 0.0260 | +0.0006 |
| 8 | `pvd_thr0p5_b0p5` | 0.5617 | -0.0023 | 0.5335 | -0.0048 | 0.6862 | +0.0062 | 0.0254 | +0.0000 |
| 9 | `pvd_thr0p5_b1p0` | 0.5591 | -0.0049 | 0.5296 | -0.0086 | 0.6839 | +0.0039 | 0.0266 | +0.0012 |
| 10 | `base` | 0.5641 | +0.0000 | 0.5383 | +0.0000 | 0.6800 | +0.0000 | 0.0254 | +0.0000 |

## Proposal Classifier Diagnostics

| label | count | proposal acc | pred background |
|---|---:|---:|---:|
| `picture` | 159 | 0.6289 | 0.3522 |
| `counter` | 237 | 0.8481 | 0.0000 |
| `desk` | 1113 | 0.7116 | 0.2668 |
| `sink` | 45 | 0.9778 | 0.0000 |
| `door` | 3098 | 0.8861 | 0.0617 |
| `shower curtain` | 0 | nan | nan |
| `cabinet` | 2384 | 0.5998 | 0.3385 |
| `table` | 4619 | 0.8303 | 0.0141 |
| `wall` | 19049 | 0.9404 | 0.0249 |
| `background` | 29327 | 0.9540 | 0.9540 |

## Interpretation Gate

- PVD go: picture improves by >=0.03 with mIoU >= base -0.002, or mIoU improves by >=0.003.
- If proposal classifier is accurate but fusion is no-go, proposal selection exists but point fusion/object mask assignment is the bottleneck.
- If proposal classifier itself is weak on picture, the proposal-first family needs stronger object proposal supervision or richer masks.

## Files

- Variant CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/proposal_verify_smoke/proposal_verify_decoder_variants.csv`
- Proposal classifier CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/proposal_verify_smoke/proposal_classifier_confusion.csv`
