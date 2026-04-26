# Fusion Protocol Alignment

This table separates the raw-point aligned diagnostic fusion protocol from the Pointcept precise/test path. The fusion rows should not be reported as official SOTA numbers until the fusion output is evaluated under the same final protocol.

| row | protocol | mIoU | mAcc | allAcc | delta vs raw fullFT |
|---|---|---:|---:|---:|---:|
| `Concerto fullFT raw-aligned single-pass` | raw-point aligned cache; one pass; no Pointcept test fragments/voting | `0.7969` | `0.8779` | `0.9276` | `+0.0000` |
| `Concerto fullFT Pointcept precise/test` | Pointcept tester from training log; model_best test path | `0.8075` | `0.8838` | `0.9309` | `+0.0106` |
| `5-expert avgprob raw-aligned` | raw-point aligned cache fusion; diagnostic, not official SOTA protocol | `0.8065` | `0.8832` | `0.9321` | `+0.0096` |
| `5-expert oracle raw-aligned` | diagnostic oracle upper bound using labels | `0.8969` | `0.9418` | `0.9684` | `+0.1000` |

Interpretation: the local full-FT reference is `0.8075` under Pointcept's test path, but `0.7969` in the raw-aligned single-pass cache protocol used by current fusion diagnostics. Current fusion gains should therefore be interpreted relative to the raw fullFT row until a final method is exported through a matched official evaluation path.
