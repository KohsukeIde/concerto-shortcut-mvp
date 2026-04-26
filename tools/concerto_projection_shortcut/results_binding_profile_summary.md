# Binding Profile Summary

This is a paper-facing consolidation table. It does not replace the source result files; it aligns train-side/readout/support quantities into one view for the central binding-profile figure.

| domain | model | task | base metric | top2/proxy | top5/proxy | pair/confusion | pair probe | random20 damage | structured20 damage | feature-zero damage |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| `scene` | `Concerto` | `ScanNet20 picture_vs_wall` | `0.4034` | `0.8929` | `0.9599` | `picture->wall (0.4382)` | `0.7041` | `` | `` | `` |
| `scene` | `Sonata` | `ScanNet20 picture_vs_wall` | `0.3582` | `0.7102` | `0.8926` | `picture->wall (0.4783)` | `0.7501` | `` | `` | `` |
| `scene` | `PTv3 supervised` | `ScanNet20 picture_vs_wall` | `0.4908` | `0.8791` | `0.9952` | `picture->wall (0.2326)` | `0.9626` | `` | `` | `` |
| `scene` | `Utonia` | `ScanNet20 picture_vs_wall` | `0.2952` | `0.9994` | `1.0000` | `picture->wall (0.1284)` | `0.8847` | `` | `` | `` |
| `scene-support` | `Concerto decoder` | `ScanNet/S3DIS full-scene support stress` | `0.7863` | `` | `` | `` | `` | `0.0336` | `0.4851` | `0.7180` |
| `scene-support` | `Concerto linear` | `ScanNet/S3DIS full-scene support stress` | `0.7696` | `` | `` | `` | `` | `0.0177` | `0.4789` | `0.7308` |
| `scene-support` | `Sonata linear` | `ScanNet/S3DIS full-scene support stress` | `0.7170` | `` | `` | `` | `` | `0.0305` | `0.4508` | `0.6563` |
| `scene-support` | `PTv3 ScanNet20` | `ScanNet/S3DIS full-scene support stress` | `0.7697` | `` | `` | `` | `` | `0.0702` | `0.5206` | `0.7429` |
| `scene-support` | `PTv3 ScanNet200` | `ScanNet/S3DIS full-scene support stress` | `0.3458` | `` | `` | `` | `` | `0.0879` | `0.2686` | `0.3440` |
| `scene-support` | `PTv3 S3DIS` | `ScanNet/S3DIS full-scene support stress` | `0.7052` | `` | `` | `` | `` | `0.2607` | `0.4170` | `0.5914` |
| `scene-support` | `Utonia` | `ScanNet20 full-scene support stress` | `0.7586` | `` | `` | `` | `` | `0.0122` | `0.4748` | `0.0120` |
| `object` | `PointGPT-S official` | `ScanObjectNN obj_bg` | `0.9105` | `0.9776` | `0.9948` | `bag->box (0.2353)` | `0.8466` | `` | `` | `` |
| `object` | `PointGPT-S no-mask` | `ScanObjectNN obj_bg` | `0.8726` | `0.9466` | `0.9914` | `pillow->bag (0.2381)` | `0.8754` | `0.3528` | `0.6558` | `0.7814` |
| `object` | `PointGPT-S no-mask order-random` | `ScanObjectNN obj_bg` | `0.8726` | `0.9587` | `0.9931` | `pillow->bag (0.2857)` | `0.8221` | `0.4423` | `0.6506` | `0.8210` |
| `object-support` | `PointGPT-S official` | `ShapeNetPart support stress` | `0.8335` | `` | `` | `` | `` | `0.1412` | `0.1922` | `0.5751` |
| `object-support` | `PointGPT-S no-mask` | `ShapeNetPart support stress` | `0.8287` | `` | `` | `` | `` | `0.1301` | `0.1682` | `0.5964` |

## Notes

- Scene readout rows use `picture_vs_wall` from the cross-model downstream audit.
- Scene-support rows use full-scene nearest-neighbor scoring from the masking battery.
- Utonia support stress uses the released Utonia inference/head path. The default Utonia transform explicitly builds feat=(coord,color,normal); its low feature-zero damage should therefore be treated as an audited robustness/anomaly, not as evidence that the input path omits raw features.
- Object rows use ScanObjectNN `obj_bg` readout and support-stress audits.
