# ScanNet Linear Proxy Results

Generated: 2026-04-12 JST

## Finished Runs

| experiment | final eval mIoU | final eval mAcc | final eval allAcc | train best mIoU | eval count |
| --- | ---: | ---: | ---: | ---: | ---: |
| scannet-proxy-concerto-continue-lin | 0.4794 | 0.6114 | 0.7718 | 0.4552 | 11 |
| scannet-proxy-coord-mlp-continue-lin | 0.4064 | 0.5503 | 0.7422 | 0.3829 | 11 |

## Readout

- `coord_mlp` is below original continuation by 0.0730 final-eval mIoU.
- `coord_mlp` retains 84.8% of the original continuation final-eval mIoU.
- This is a soft-go signal for downstream shortcut relevance, not a full replacement result.
- `no-enc2d` is still pending; its continuation is running as `arkit-full-continue-no-enc2d`.
