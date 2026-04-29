# Scene DINO Support Recognizability

DINO sees actual ScanNet RGB frames. Patch labels are derived only from points retained by each 3D support condition.
This is a scene-level 2D semantic-evidence calibration, not a 3D segmentation result.

- DINO model: `facebook/dinov2-with-registers-giant`
- train images max: `256`
- val images max: `128`
- structured cell size: `1.28` m

| condition | patch acc | macro acc | delta macro vs clean | val samples | picture | wall | door | cabinet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 0.6207 | 0.5364 | 0.0000 | 2895 | 0.5455 | 0.6277 | 0.6730 | 0.5392 |
| `random_keep80` | 0.6162 | 0.5466 | 0.0101 | 2892 | 0.6613 | 0.6595 | 0.4654 | 0.5490 |
| `random_keep50` | 0.6091 | 0.5359 | -0.0005 | 2865 | 0.6562 | 0.6453 | 0.3919 | 0.5446 |
| `random_keep20` | 0.5589 | 0.5084 | -0.0280 | 2589 | 0.8205 | 0.6218 | 0.3719 | 0.3763 |
| `random_keep10` | 0.5601 | 0.5412 | 0.0048 | 1912 | 0.3077 | 0.6208 | 0.3636 | 0.4493 |
| `structured_keep80` | 0.6176 | 0.5635 | 0.0270 | 2764 | 0.8333 | 0.6785 | 0.4110 | 0.6458 |
| `structured_keep50` | 0.5973 | 0.4945 | -0.0419 | 2230 | 0.3333 | 0.7116 | 0.3444 | 0.5606 |
| `structured_keep20` | 0.5882 | 0.4195 | -0.1170 | 1122 | 0.4167 | 0.7516 | 0.4464 | 0.7222 |
| `structured_keep10` | 0.5644 | 0.4665 | -0.0699 | 916 | 0.8333 | 0.6527 | 0.0769 | 0.4583 |
| `instance_keep20` | 0.5201 | 0.4467 | -0.0898 | 773 | 0.6667 | 0.5969 | 0.2500 | 0.5135 |

## Paper-safe interpretation

- If a condition remains close to clean here, the corresponding RGB patches still carry semantic evidence under the stressed 3D support selection.
- If a condition drops strongly here, it should not be used to claim that the 3D model missed obvious 2D evidence.
- Because DINO sees full RGB patches, this calibrates 2D semantic evidence at retained-support locations; it is not an occluded-image VLM study.
