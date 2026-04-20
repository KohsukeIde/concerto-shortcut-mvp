# Proposal-then-Verify Decoder Pilot

Lightweight PVD pilot: learn a hard-class proposal verifier on fine voxel proposals and fuse conservative proposal boosts into base decoder logits.

## Setup

- Config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100.py`
- Weight: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`
- Region voxel size: `4`
- Positive purity: `0.8`
- Proposal classes: `picture,counter,desk,sink,door,shower curtain,cabinet,table,wall`
- Train batches: `256`
- Train proposal count: `279347`
- Seen val batches: `312`

## Headline

- Base: mIoU=0.7789, picture=0.4061, picture->wall=0.4401
- Best mIoU: `base` mIoU=0.7789 (Δ+0.0000), picture=0.4061 (Δ+0.0000), p->wall=0.4401 (Δ+0.0000)
- Best safe picture: `base` mIoU=0.7789 (Δ+0.0000), picture=0.4061 (Δ+0.0000), p->wall=0.4401 (Δ+0.0000)

## Top Variants

| rank | variant | mIoU | ΔmIoU | weak mIoU | Δweak | picture | Δpicture | p->wall | Δp->wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `pvd_thr0p9_b0p25` | 0.7784 | -0.0005 | 0.6797 | -0.0015 | 0.3952 | -0.0109 | 0.4024 | -0.0377 |
| 2 | `pvd_thr0p7_b0p25` | 0.7781 | -0.0008 | 0.6787 | -0.0025 | 0.3939 | -0.0122 | 0.3929 | -0.0472 |
| 3 | `pvd_thr0p5_b0p25` | 0.7775 | -0.0014 | 0.6773 | -0.0039 | 0.3935 | -0.0126 | 0.3911 | -0.0490 |
| 4 | `pvd_thr0p9_b0p5` | 0.7761 | -0.0029 | 0.6739 | -0.0073 | 0.3729 | -0.0332 | 0.3896 | -0.0505 |
| 5 | `pvd_thr0p7_b0p5` | 0.7743 | -0.0046 | 0.6692 | -0.0120 | 0.3603 | -0.0457 | 0.3690 | -0.0712 |
| 6 | `pvd_thr0p9_b1p0` | 0.7728 | -0.0061 | 0.6665 | -0.0147 | 0.3551 | -0.0510 | 0.3852 | -0.0549 |
| 7 | `pvd_thr0p5_b0p5` | 0.7726 | -0.0063 | 0.6653 | -0.0159 | 0.3565 | -0.0496 | 0.3623 | -0.0778 |
| 8 | `pvd_thr0p7_b1p0` | 0.7674 | -0.0116 | 0.6529 | -0.0283 | 0.3117 | -0.0944 | 0.3487 | -0.0914 |
| 9 | `pvd_thr0p5_b1p0` | 0.7631 | -0.0158 | 0.6427 | -0.0385 | 0.2892 | -0.1169 | 0.3279 | -0.1122 |
| 10 | `base` | 0.7789 | +0.0000 | 0.6812 | +0.0000 | 0.4061 | +0.0000 | 0.4401 | +0.0000 |

## Proposal Classifier Diagnostics

| label | count | proposal acc | pred background |
|---|---:|---:|---:|
| `picture` | 13438 | 0.6587 | 0.0209 |
| `counter` | 16287 | 0.9083 | 0.0014 |
| `desk` | 49926 | 0.8895 | 0.0271 |
| `sink` | 6233 | 0.8702 | 0.0026 |
| `door` | 101667 | 0.9414 | 0.0142 |
| `shower curtain` | 8854 | 0.9136 | 0.0705 |
| `cabinet` | 104197 | 0.8786 | 0.0507 |
| `table` | 115687 | 0.8833 | 0.0369 |
| `wall` | 764621 | 0.8859 | 0.0331 |
| `background` | 1406698 | 0.9328 | 0.9328 |

## Interpretation Gate

- PVD go: picture improves by >=0.03 with mIoU >= base -0.002, or mIoU improves by >=0.003.
- If proposal classifier is accurate but fusion is no-go, proposal selection exists but point fusion/object mask assignment is the bottleneck.
- If proposal classifier itself is weak on picture, the proposal-first family needs stronger object proposal supervision or richer masks.

## Files

- Variant CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/proposal_verify_decoder/proposal_verify_decoder_variants.csv`
- Proposal classifier CSV: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/scannet_decoder_probe_origin/proposal_verify_decoder/proposal_classifier_confusion.csv`
