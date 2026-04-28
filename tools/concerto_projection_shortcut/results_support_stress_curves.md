# Support-Stress Curves

This file normalizes scene and object support-stress results into a single schema.
`B_down` is `(clean_score - stress_score) / (clean_score - null_score)`, where the null row is feature-zero for scene models and xyz-zero for object models when available.

## Outputs

- Unified CSV: `tools/concerto_projection_shortcut/results_support_stress_curves.csv`
- Keep20 ratio CSV: `tools/concerto_projection_shortcut/results_support_stress_keep20_ratios.csv`
- Figure PNG: `tools/concerto_projection_shortcut/results_support_stress_curves.png`
- Figure PDF: `tools/concerto_projection_shortcut/results_support_stress_curves.pdf`

## Keep20 Stress Ratios

| domain | model | task | random20 damage | structured/random | object-or-part/random |
|---|---|---|---:|---:|---:|
| `object` | `PointGPT-S mask-on order-random` | `ScanObjectNN obj_bg` | 0.3046 | 2.08 | -- |
| `object` | `PointGPT-S no-mask` | `ScanObjectNN obj_bg` | 0.3804 | 1.74 | -- |
| `object` | `PointGPT-S no-mask` | `ShapeNetPart` | 0.1271 | 1.37 | 2.81 |
| `object` | `PointGPT-S no-mask order-random` | `ScanObjectNN obj_bg` | 0.4509 | 1.46 | -- |
| `object` | `PointGPT-S official` | `ScanObjectNN obj_bg` | 0.4768 | 1.47 | -- |
| `object` | `PointGPT-S official` | `ShapeNetPart` | 0.1407 | 1.34 | 2.48 |
| `scene` | `Concerto-D` | `ScanNet20` | 0.0342 | 14.37 | 17.45 |
| `scene` | `Concerto-L` | `ScanNet20` | 0.0180 | 26.03 | 31.85 |
| `scene` | `PTv3` | `S3DIS Area-5` | 0.2661 | 1.69 | 2.22 |
| `scene` | `PTv3` | `ScanNet20` | 0.0750 | 6.95 | 8.44 |
| `scene` | `PTv3` | `ScanNet200` | 0.0829 | 3.09 | 3.61 |
| `scene` | `Sonata-L` | `ScanNet20` | 0.0297 | 14.81 | 18.08 |
| `scene` | `Utonia` | `ScanNet20` | 0.0111 | 42.06 | 47.98 |

## Interpretation

- Random keep is consistently a weaker stress than structured or object/part-aware removal in the same task family.
- This supports the scoped claim that random point-drop robustness is an insufficient robustness test.
- These curves do not by themselves prove an architecture-level cause; architecture claims require grouping or patchization ablations.
