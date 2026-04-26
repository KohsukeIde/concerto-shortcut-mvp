# Train-Split FullFT-Centered Residual Fusion

This is the publishable-protocol pilot: residual fusion is trained on ScanNet train scenes, selected on held-out train scenes, and evaluated once on ScanNet val. It should not be mixed with the earlier val-CV diagnostic.

- experts: Concerto fullFT default + Concerto decoder, Sonata linear, Utonia
- max train scenes: `384`
- train/heldout points: `628736` / `315392`
- val scenes: `312`
- feature pairs: `True`; projection dim `64`
- selected by heldout mIoU: `train_residual_kl0.03_safe4`

## Heldout Selection

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `train_residual_kl0.03_safe4` | `0.9458` | `0.9786` | `0.9173` | `0.8720` | `0.0389` | `0.8988` | `0.9585` | `0.9216` |
| 2 | `train_residual_kl0_safe4` | `0.9457` | `0.9785` | `0.9179` | `0.8702` | `0.0368` | `0.9012` | `0.9579` | `0.9205` |
| 3 | `train_residual_kl0_safe2` | `0.9452` | `0.9782` | `0.9169` | `0.8681` | `0.0341` | `0.9032` | `0.9567` | `0.9178` |
| 4 | `train_residual_kl0.03_safe2` | `0.9450` | `0.9783` | `0.9160` | `0.8648` | `0.0341` | `0.9011` | `0.9561` | `0.9191` |

## Final Val Evaluation

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `avgprob_all` | `0.8023` | `0.9302` | `0.7079` | `0.4171` | `0.4579` | `0.7263` | `0.7687` | `0.7899` |
| 2 | `single::Concerto fullFT` | `0.7960` | `0.9276` | `0.7001` | `0.4313` | `0.4009` | `0.7089` | `0.7773` | `0.7774` |
| 3 | `train_residual_kl0.03_safe4` | `0.7960` | `0.9276` | `0.6992` | `0.4278` | `0.4228` | `0.7066` | `0.7781` | `0.7762` |
