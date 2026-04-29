# Scene-Level 2D Recognizability Claim Line

This note summarizes how to use `results_scene_dino_support_recognizability.md` in the paper.

## Setup

- Model: `facebook/dinov2-with-registers-giant`.
- Input to DINO: actual ScanNet RGB frames.
- Stress protocol: DINO features are unchanged, but patch labels/evaluation samples are derived only from 3D points retained by each support condition.
- This is a 2D semantic-evidence calibration for retained 3D support, not an occluded-image VLM or segmentation result.
- Conditions use the same sampled RGB frame order across clean/random/structured/instance variants.

## Key Values

| condition | DINO patch acc | DINO macro acc | macro / clean | val patches |
|---|---:|---:|---:|---:|
| clean | 0.6207 | 0.5364 | 1.000 | 2895 |
| random_keep80 | 0.6162 | 0.5466 | 1.019 | 2892 |
| random_keep50 | 0.6091 | 0.5359 | 0.999 | 2865 |
| random_keep20 | 0.5589 | 0.5084 | 0.948 | 2589 |
| random_keep10 | 0.5601 | 0.5412 | 1.009 | 1912 |
| structured_keep80 | 0.6176 | 0.5635 | 1.050 | 2764 |
| structured_keep50 | 0.5973 | 0.4945 | 0.922 | 2230 |
| structured_keep20 | 0.5882 | 0.4195 | 0.782 | 1122 |
| structured_keep10 | 0.5644 | 0.4665 | 0.870 | 916 |
| instance_keep20 | 0.5201 | 0.4467 | 0.833 | 773 |

## Paper-Safe Use

Use:

- Random retained support is a weak scene-level stress: DINO macro accuracy remains essentially clean at keep80/keep50 and still retains about 95% of clean macro accuracy at keep20.
- Structured and instance-aware support removal are semantically harsher than random drop, both in retained patch count and DINO macro accuracy.
- This supports the paper's scoped claim that random point-drop robustness is not sufficient evidence of target-relevant robustness.

Do not use:

- Do not claim structured_keep20/10 are obviously solvable from 2D evidence. DINO macro drops to 0.4195/0.4665 and retained patch counts are much smaller.
- Do not claim this proves a 3D architecture failure. It calibrates the semantic evidence left by support stresses; architecture-level claims still require grouping/patchization ablations.
- Do not claim this is a human or VLM judgement. It is a DINO patch-probe calibration aligned with Concerto's DINO-based objective.

## Suggested Sentence

To calibrate whether support stresses preserve visible semantic evidence, we train condition-specific DINO patch probes on ScanNet RGB frames while deriving patch labels only from retained 3D support. Random support removal leaves the DINO probe nearly unchanged through keep50 and retains 95% of clean macro accuracy at keep20, whereas structured and instance-aware removal substantially reduce retained patch coverage and macro accuracy. Thus random point-drop is a weak stress, while structured/object-style support removal should be interpreted as a genuinely harder missing-support condition rather than an obvious-to-2D failure case.
