# ScanNet Linear Proxy Results

Updated: 2026-04-14 16:10 JST

## Runs

| experiment | status | final/last eval mIoU | final/last eval mAcc | final/last eval allAcc | train best mIoU | eval count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| scannet-proxy-concerto-continue-lin | finished | 0.4794 | 0.6114 | 0.7718 | 0.4552 | 11 |
| scannet-proxy-coord-mlp-continue-lin | finished | 0.4064 | 0.5503 | 0.7422 | 0.3829 | 11 |
| scannet-proxy-no-enc2d-continue-lin | finished | 0.4010 | 0.5440 | 0.7391 | 0.3765 | 11 |
| scannet-proxy-no-enc2d-renorm-continue-lin | finished validation / full test aborted (disk) | 0.3794 | 0.5510 | 0.7282 | 0.3802 | 10 |

## Readout

- `coord_mlp` is below original continuation by 0.0730 final/last-eval mIoU.
- `coord_mlp` retains 84.8% of the original continuation final/last-eval mIoU.
- `no-enc2d` linear is now finished.
- On final/last eval mIoU, `coord_mlp` is +0.0054 above `no-enc2d` and -0.0730 below original.
- On training-side best mIoU, `coord_mlp` is +0.0064 above `no-enc2d` and -0.0723 below original.
- This means `coord_mlp` recovers 6.9% of the `no-enc2d` -> original gap by final/last eval mIoU and 8.1% by training-side best mIoU.
- `no-enc2d-renorm` finished linear training and validation, but its post-train full test aborted during logging with `No space left on device`.
- Relative to `no-enc2d-renorm`, `coord_mlp` is +0.0270 final/last-eval mIoU and +0.0027 training-side best mIoU.
- Relative to `no-enc2d-renorm`, `coord_mlp` recovers 27.0% of the gap to original by final/last eval mIoU, but only 3.6% by training-side best mIoU.
- Current interpretation: downstream shortcut relevance remains measurable, but the renormalized control further weakens the stronger "mostly coordinate shortcut" claim in this continuation proxy. The objective-level shortcut claim is unchanged.
