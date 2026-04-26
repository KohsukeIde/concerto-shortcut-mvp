# Train-Split FullFT-Centered Residual Fusion

This is the publishable-protocol pilot: residual fusion is trained on ScanNet train scenes, selected on held-out train scenes, and evaluated once on ScanNet val. It should not be mixed with the earlier val-CV diagnostic.

- experts: Concerto fullFT default + Concerto decoder, Sonata linear, Utonia, PTv3_supervised
- max train scenes: `384`
- train/heldout points: `628736` / `315392`
- val scenes: `312`
- feature pairs: `True`; projection dim `64`
- selected by heldout mIoU: `train_residual_kl0_safe2`

## Heldout Selection

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `train_residual_kl0_safe2` | `0.9468` | `0.9788` | `0.9192` | `0.8690` | `0.0362` | `0.9065` | `0.9590` | `0.9212` |
| 2 | `train_residual_kl0.03_safe4` | `0.9465` | `0.9789` | `0.9181` | `0.8716` | `0.0352` | `0.9005` | `0.9589` | `0.9234` |
| 3 | `train_residual_kl0.03_safe2` | `0.9463` | `0.9787` | `0.9181` | `0.8713` | `0.0378` | `0.9032` | `0.9590` | `0.9228` |
| 4 | `train_residual_kl0_safe4` | `0.9462` | `0.9789` | `0.9174` | `0.8696` | `0.0326` | `0.9006` | `0.9580` | `0.9228` |

## Final Val Evaluation

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `avgprob_all` | `0.8061` | `0.9319` | `0.7115` | `0.4258` | `0.4607` | `0.7241` | `0.7706` | `0.7901` |
| 2 | `train_residual_kl0_safe2` | `0.7976` | `0.9280` | `0.7021` | `0.4333` | `0.4113` | `0.7100` | `0.7739` | `0.7756` |
| 3 | `single::Concerto fullFT` | `0.7967` | `0.9276` | `0.7009` | `0.4337` | `0.3929` | `0.7100` | `0.7719` | `0.7753` |
