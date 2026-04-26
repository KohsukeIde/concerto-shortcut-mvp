# Cross-Model FullFT-Centered Residual Fusion Diagnostic

Two-fold scene-level validation pilot. This uses ScanNet val labels only to test whether a fullFT-centered residual fusion decoder is a plausible method direction; it is not a final train-split result.

- experts: Concerto fullFT default + Concerto decoder, Sonata linear, Utonia, PTv3_supervised
- sampled train points per fold: [8192, 8192]
- feature projection dim: `64`
- epochs: `2`
- KL weights: `0.0,0.03,0.1`
- safe CE weights: `1.0,2.0,4.0`

| rank | variant | mIoU | allAcc | weak mIoU | picture | p->wall | counter | cabinet | door |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `single::Concerto fullFT` | `0.4655` | `0.9031` | `0.3638` | `0.9389` | `0.0159` | `0.0000` | `0.2871` | `0.8566` |
| 2 | `cv_residual_kl0.1_safe4` | `0.4653` | `0.9028` | `0.3640` | `0.9437` | `0.0165` | `0.0000` | `0.2850` | `0.8551` |
| 3 | `cv_residual_kl0_safe1` | `0.4653` | `0.9028` | `0.3638` | `0.9421` | `0.0165` | `0.0000` | `0.2855` | `0.8552` |
| 4 | `cv_residual_kl0.03_safe1` | `0.4653` | `0.9028` | `0.3640` | `0.9437` | `0.0165` | `0.0000` | `0.2848` | `0.8553` |
| 5 | `cv_residual_kl0_safe2` | `0.4652` | `0.9028` | `0.3639` | `0.9426` | `0.0165` | `0.0000` | `0.2852` | `0.8551` |
| 6 | `cv_residual_kl0.03_safe2` | `0.4652` | `0.9029` | `0.3638` | `0.9416` | `0.0165` | `0.0000` | `0.2854` | `0.8555` |
| 7 | `cv_residual_kl0.1_safe2` | `0.4652` | `0.9028` | `0.3637` | `0.9410` | `0.0165` | `0.0000` | `0.2852` | `0.8553` |
| 8 | `cv_residual_kl0_safe4` | `0.4652` | `0.9028` | `0.3637` | `0.9416` | `0.0165` | `0.0000` | `0.2849` | `0.8554` |
| 9 | `cv_residual_kl0.1_safe1` | `0.4652` | `0.9028` | `0.3638` | `0.9421` | `0.0165` | `0.0000` | `0.2849` | `0.8553` |
| 10 | `cv_residual_kl0.03_safe4` | `0.4651` | `0.9028` | `0.3636` | `0.9410` | `0.0165` | `0.0000` | `0.2847` | `0.8553` |
| 11 | `avgprob_all` | `0.4165` | `0.8945` | `0.2318` | `0.0945` | `0.8991` | `0.0000` | `0.1966` | `0.8685` |
