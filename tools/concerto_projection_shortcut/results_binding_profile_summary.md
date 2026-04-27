# Binding Profile Summary

This is a paper-facing consolidation table. It does not replace the source result files; it aligns train-side/readout/support quantities into one view for the central binding-profile figure.

| domain | model | task | base metric | top2/proxy | top5/proxy | pair/confusion | pair probe | random20 damage | structured20 damage | feature-zero damage |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| `scene` | `Concerto` | `ScanNet20 picture_vs_wall` | `0.4034` | `0.8929` | `0.9599` | `picture->wall (0.4382)` | `0.7041` | `` | `` | `` |
| `scene` | `Sonata` | `ScanNet20 picture_vs_wall` | `0.3582` | `0.7102` | `0.8926` | `picture->wall (0.4783)` | `0.7501` | `` | `` | `` |
| `scene` | `PTv3 supervised` | `ScanNet20 picture_vs_wall` | `0.4908` | `0.8791` | `0.9952` | `picture->wall (0.2326)` | `0.9626` | `` | `` | `` |
| `scene` | `Utonia` | `ScanNet20 picture_vs_wall` | `0.2952` | `0.9994` | `1.0000` | `picture->wall (0.1284)` | `0.8847` | `` | `` | `` |
| `scene-support` | `Concerto decoder` | `ScanNet20 support-stress severity` | `0.7864` | `` | `` | `` | `` | `0.0342` | `0.4920` | `0.7183` |
| `scene-support` | `Concerto linear` | `ScanNet20 support-stress severity` | `0.7696` | `` | `` | `` | `` | `0.0180` | `0.4688` | `0.7305` |
| `scene-support` | `Sonata linear` | `ScanNet20 support-stress severity` | `0.7164` | `` | `` | `` | `` | `0.0297` | `0.4403` | `0.6558` |
| `scene-support` | `PTv3 ScanNet20` | `ScanNet20 support-stress severity` | `0.7713` | `` | `` | `` | `` | `0.0750` | `0.5212` | `0.7442` |
| `scene-support` | `PTv3 ScanNet200` | `ScanNet200 support-stress severity` | `0.3420` | `` | `` | `` | `` | `0.0829` | `0.2562` | `0.3401` |
| `scene-support` | `PTv3 S3DIS` | `S3DIS Area-5 support-stress severity` | `0.7112` | `` | `` | `` | `` | `0.2661` | `0.4486` | `0.5966` |
| `scene-support` | `Utonia` | `ScanNet20 full-scene support stress` | `0.7580` | `` | `` | `` | `` | `0.0111` | `0.4680` | `0.0103` |
| `object` | `PointGPT-S official` | `ScanObjectNN obj_bg` | `0.9105` | `0.9776` | `0.9948` | `bag->box (0.2353)` | `0.8466` | `0.4768` | `0.7005` | `0.8176` |
| `object` | `PointGPT-S no-mask` | `ScanObjectNN obj_bg` | `0.8726` | `0.9466` | `0.9914` | `pillow->bag (0.2381)` | `0.8754` | `0.3804` | `0.6609` | `0.7986` |
| `object` | `PointGPT-S no-mask order-random` | `ScanObjectNN obj_bg` | `0.8726` | `0.9587` | `0.9931` | `pillow->bag (0.2857)` | `0.8221` | `0.4509` | `0.6592` | `0.8021` |
| `object` | `PointGPT-S mask-on order-random` | `ScanObjectNN obj_bg` | `0.8795` | `0.9552` | `0.9983` | `bed->sofa (0.1818)` | `0.8972` | `0.3046` | `0.6334` | `0.8124` |
| `object-support` | `PointGPT-S official` | `ShapeNetPart support stress` | `0.8335` | `` | `` | `` | `` | `0.1407` | `0.1882` | `0.5747` |
| `object-support` | `PointGPT-S no-mask` | `ShapeNetPart support stress` | `0.8287` | `` | `` | `` | `` | `0.1271` | `0.1741` | `0.5960` |

## Notes

- Scene readout rows use `picture_vs_wall` from the cross-model downstream audit.
- Scene-support rows use full-scene nearest-neighbor scoring from the masking battery.
- Utonia support stress uses the released Utonia inference/head path. The default Utonia transform explicitly builds feat=(coord,color,normal); its low feature-zero damage should therefore be treated as an audited robustness/anomaly, not as evidence that the input path omits raw features.
- Object rows use ScanObjectNN `obj_bg` readout and support-stress audits.
