# CoDA Decoder Adapter

## Summary

This is the first lightweight decoder-adaptation pilot after the fixed-logit
reranker family failed to recover meaningful oracle headroom.

The base checkpoint is the same `concerto_base_origin.pth` ScanNet decoder-probe
checkpoint used for the origin decoder stage-wise and oracle/actionability
analysis. The Concerto encoder/decoder feature extractor is frozen. CoDA trains
only a small residual adapter:

```text
z = z0 + A(h)
```

where `h` is the frozen decoder point feature and `z0` is the base 20-way
decoder logit. The loss combines weighted 20-way CE, confusion-pair auxiliary
CE, KL to the base logits, and residual L2 regularization.

Result: **no-go as a paper-relevant positive**.

The adapter is trainable and improves the train/heldout objective, but the
heldout-selected setting overcorrects ScanNet val. A follow-up all-variant val
sweep finds only a tiny positive.

Best ScanNet val movement:

- best mIoU variant: `lam0p1_tau1`
  - mIoU: `0.77769963 -> 0.77793617`, delta `+0.00023654`
  - `picture` IoU: `0.40359095 -> 0.40478069`, delta `+0.00118973`
  - `picture -> wall`: `0.44355049 -> 0.43346092`
- best safe `picture` movement: `lam0p2_tau1`
  - mIoU: `0.77769963 -> 0.77788432`, delta `+0.00018469`
  - `picture` IoU: `0.40359095 -> 0.40526737`, delta `+0.00167642`
  - `picture -> wall`: `0.44355049 -> 0.42317853`

This is directionally consistent with the oracle/actionability result, but far
below the planned gate.

## Setup

Base checkpoint:

- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/model/model_best.pth`

Config:

- `data/runs/scannet_decoder_probe_origin/exp/scannet-dec-origin-e100/config.py`

Implementation:

- `tools/concerto_projection_shortcut/fit_coda_decoder_adapter.py`
- `tools/concerto_projection_shortcut/submit_coda_decoder_adapter_abciq_qf.sh`

Run outputs:

- `data/runs/scannet_decoder_probe_origin/coda_decoder_adapter_fullscan/coda_decoder_adapter.md`
- `data/runs/scannet_decoder_probe_origin/coda_decoder_adapter_fullscan/coda_heldout_sweep.csv`
- `data/runs/scannet_decoder_probe_origin/coda_decoder_adapter_fullscan/coda_val_selected.csv`
- `data/runs/scannet_decoder_probe_origin/coda_decoder_adapter_fullscan/coda_adapter.pt`
- `data/runs/scannet_decoder_probe_origin/coda_decoder_adapter_val_all/coda_val_all.md`
- `data/runs/scannet_decoder_probe_origin/coda_decoder_adapter_val_all/coda_val_all.csv`
- `data/logs/abciq/133361.qjcm.OU`
- `data/logs/abciq/133362.qjcm.OU`

## Jobs

| run | job | resource | walltime used | status | note |
| --- | --- | --- | ---: | --- | --- |
| CoDA train + heldout select + selected val | `133361.qjcm` | `rt_QF=1` | `00:03:02` | `Exit_status=0` | selected `lam1_tau1`, which overcorrects val |
| saved-adapter all-variant val sweep | `133362.qjcm` | `rt_QF=1` | `00:00:57` | `Exit_status=0` | checks small-correction variants without retraining |

## Training

Full-scan collection:

- train points: `1200000`
- heldout train points: `1168197`
- train scenes scanned: `1201`
- trainable adapter parameters: `54184`

Training trace:

| step | loss | CE | pair | KL | delta L2 | base acc | adapter acc | delta RMS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2827 | 0.2293 | 0.0952 | 0.0584 | 0.2862 | 0.9500 | 0.9419 | 0.5350 |
| 300 | 0.1720 | 0.1373 | 0.0555 | 0.0942 | 0.2236 | 0.9484 | 0.9670 | 0.4729 |
| 600 | 0.1541 | 0.1229 | 0.0485 | 0.0965 | 0.2051 | 0.9492 | 0.9711 | 0.4529 |
| 900 | 0.1584 | 0.1273 | 0.0489 | 0.0944 | 0.1939 | 0.9471 | 0.9668 | 0.4403 |
| 1200 | 0.1443 | 0.1175 | 0.0406 | 0.0936 | 0.1850 | 0.9471 | 0.9695 | 0.4302 |
| 1500 | 0.1756 | 0.1396 | 0.0588 | 0.0962 | 0.1820 | 0.9460 | 0.9688 | 0.4266 |
| 1800 | 0.1622 | 0.1307 | 0.0501 | 0.0938 | 0.1734 | 0.9491 | 0.9678 | 0.4164 |
| 2100 | 0.1528 | 0.1230 | 0.0465 | 0.0951 | 0.1752 | 0.9453 | 0.9698 | 0.4186 |
| 2400 | 0.1705 | 0.1397 | 0.0490 | 0.0930 | 0.1681 | 0.9464 | 0.9669 | 0.4100 |
| 2700 | 0.1781 | 0.1424 | 0.0586 | 0.0947 | 0.1663 | 0.9446 | 0.9695 | 0.4078 |
| 3000 | 0.1650 | 0.1325 | 0.0524 | 0.0935 | 0.1627 | 0.9498 | 0.9705 | 0.4034 |

The adapter can fit the train minibatch objective: adapter accuracy rises from
roughly `0.95` base to `0.97`. This did not transfer to a useful ScanNet val
gain.

## Heldout Selection

The heldout train split selects the most aggressive variant:

| variant | mIoU | delta | picture IoU | picture delta | picture->wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.9097 | +0.0000 | 0.9186 | +0.0000 | 0.0736 |
| `lam1_tau1` | 0.9252 | +0.0156 | 0.9430 | +0.0244 | 0.0445 |

But on ScanNet val this overcorrects:

| variant | mIoU | delta | picture IoU | picture delta | picture->wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.7788 | +0.0000 | 0.4093 | +0.0000 | 0.4337 |
| `lam1_tau1` | 0.7674 | -0.0114 | 0.3810 | -0.0283 | 0.3430 |

This is a strong train/heldout-to-val transfer warning.

## ScanNet Val All-Variant Sweep

The saved adapter was then evaluated on all lambda/tau variants without
retraining.

| variant | mIoU | delta | picture IoU | picture delta | picture->wall | counter delta | desk delta | sink delta | cabinet delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.7777 | +0.0000 | 0.4036 | +0.0000 | 0.4436 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| `lam0p1_tau1` | 0.7779 | +0.0002 | 0.4048 | +0.0012 | 0.4335 | +0.0030 | +0.0002 | -0.0012 | +0.0003 |
| `lam0p2_tau1` | 0.7779 | +0.0002 | 0.4053 | +0.0017 | 0.4232 | +0.0056 | -0.0005 | -0.0019 | +0.0003 |
| `lam0p5_tau0p5` | 0.7776 | -0.0001 | 0.4049 | +0.0013 | 0.4179 | +0.0061 | -0.0014 | -0.0022 | -0.0000 |
| `lam1_tau1` | 0.7664 | -0.0113 | 0.3775 | -0.0261 | 0.3524 | +0.0026 | -0.0263 | -0.0170 | -0.0081 |

Small corrections are weakly positive; stronger corrections reduce
`picture -> wall` but hurt `picture` IoU and overall mIoU.

## Interpretation

- CoDA changes the method family relative to fixed-logit reranking: the adapter
  is a trainable feature-to-logit residual map.
- It still does not recover meaningful oracle headroom.
- The result strengthens the read that the oracle gap is not recoverable by a
  simple offline post-hoc adapter trained on cached decoder features.
- The next positive attempt should either:
  - train the decoder/adaptation inside the original Pointcept training loop
    with real augmentation and validation-aware early stopping;
  - use a more constrained class-prior / calibration objective targeted only at
    rare classes;
  - or move beyond same-checkpoint readout-side fixes.

## Decision

**No-go for current CoDA pilot.**

Do not spend more runs on cached-feature adapters of this exact form unless the
training protocol changes materially, e.g. in-loop decoder adaptation with
augmentation or a stricter rare-class calibration constraint.
