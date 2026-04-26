# Cross-Model FullFT-Centered Residual Fusion Diagnostic

Two-fold scene-level validation pilot. This uses ScanNet val labels only to test whether a fullFT-centered residual fusion decoder is a plausible method direction; it is not a final train-split result.

- experts: Concerto fullFT default + Concerto decoder, Sonata linear, Utonia, PTv3_supervised
- sampled train points per fold: [159744, 159744]
- feature projection dim: `64`
- epochs: `10`
- KL weights: `0.0`
- safe CE weights: `1.0,4.0`

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `cv_residual_kl0_safe4` | `0.8164` | `0.9360` | `0.7092` | `0.3999` | `0.4808` | `0.6914` | `0.7681` | `0.8033` |
| 2 | `avgprob_all` | `0.8065` | `0.9320` | `0.7109` | `0.4235` | `0.4626` | `0.7232` | `0.7725` | `0.7889` |
| 3 | `cv_residual_kl0_safe1` | `0.8058` | `0.9320` | `0.6883` | `0.3811` | `0.4927` | `0.6364` | `0.7525` | `0.7897` |
| 4 | `single::Concerto fullFT` | `0.7969` | `0.9277` | `0.6991` | `0.4349` | `0.3983` | `0.7049` | `0.7769` | `0.7724` |
