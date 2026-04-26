# Cross-Model FullFT-Centered Residual Fusion Diagnostic

Two-fold scene-level validation pilot. This uses ScanNet val labels only to test whether a fullFT-centered residual fusion decoder is a plausible method direction; it is not a final train-split result.

- experts: Concerto fullFT default + Concerto decoder, Sonata linear, Utonia, PTv3_supervised
- sampled train points per fold: [319488, 319488]
- feature projection dim: `64`
- epochs: `20`
- KL weights: `0.03`
- safe CE weights: `2.0`

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `cv_residual_kl0.03_safe2` | `0.8097` | `0.9345` | `0.6957` | `0.3821` | `0.5072` | `0.6675` | `0.7644` | `0.7983` |
| 2 | `avgprob_all` | `0.8065` | `0.9320` | `0.7109` | `0.4235` | `0.4626` | `0.7232` | `0.7725` | `0.7889` |
| 3 | `single::Concerto fullFT` | `0.7969` | `0.9277` | `0.6991` | `0.4349` | `0.3983` | `0.7049` | `0.7769` | `0.7724` |
