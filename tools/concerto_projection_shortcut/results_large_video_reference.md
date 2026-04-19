# Large-Video Checkpoint Reference Results

## Scope

These results use the released large-video Concerto checkpoint:

- `data/weights/concerto/pretrain-concerto-v1m1-2-large-video.pth`

They are retained as **cross-variant evidence**. They should not be mixed into
mainline claims about the Concerto paper main/origin variant
`concerto_base_origin.pth` unless explicitly labelled as cross-variant
comparison.

## Why Keep These Results

- The large-video checkpoint is the released full checkpoint that includes
  enc2d heads, so it allows an exact released-checkpoint causal battery without
  frozen-backbone head refit.
- It provides useful evidence that the shortcut signal appears in a released
  Concerto-family model beyond the ARKit continuation regime.
- It is not the Concerto paper origin checkpoint, so it is secondary for the
  main paper claim.

## Result Files

| result | file | role |
| --- | --- | --- |
| Official large-video ARKit/ScanNet causal battery | `tools/concerto_projection_shortcut/results_official_causal_battery.md` | cross-variant shortcut evidence |
| SR-LoRA v5 Phase A on large-video | `tools/concerto_projection_shortcut/results_sr_lora_phasea.md` | negative intervention result on released full checkpoint |
| ScanNet class-wise diagnosis on large-video linear/SR-LoRA | `tools/concerto_projection_shortcut/results_scannet_classwise_diagnosis.md` | class/confusion analysis for large-video line |
| ScanNet point-stage trace on large-video linear | `tools/concerto_projection_shortcut/results_scannet_point_stagewise_trace.md` | legacy point-stage trace for large-video line |
| ScanNet counterfactual downstream on large-video linear | `tools/concerto_projection_shortcut/results_scannet_counterfactual_downstream.md` | downstream coordinate dependence diagnostic |
| DINO / Concerto patch separation using large-video config | `tools/concerto_projection_shortcut/results_concerto3d_patch_separation_stepA.md` | first-pass patch-level diagnostic |
| Exact DINO / Concerto patch controls | `tools/concerto_projection_shortcut/results_concerto3d_dino_exact_controls_stepA.md` | corrected apples-to-apples patch subset control |
| Patch-stage trace | `tools/concerto_projection_shortcut/results_concerto3d_stagewise_trace_stepA.md` | stage-wise patch trace |

## Key Causal Battery Numbers

From `results_official_causal_battery.md`:

| dataset | mode | enc2d loss | delta vs baseline |
| --- | --- | ---: | ---: |
| ARKit val | baseline | 2.737827 | 0.000000 |
| ARKit val | global target permutation | 3.324298 | +0.586471 |
| ARKit val | cross-image target swap | 3.371361 | +0.633534 |
| ARKit val | cross-scene target swap | 3.324546 | +0.586719 |
| ScanNet val | baseline | 3.416545 | 0.000000 |
| ScanNet val | global target permutation | 5.466386 | +2.049841 |
| ScanNet val | cross-image target swap | 5.502719 | +2.086174 |
| ScanNet val | cross-scene target swap | 5.467065 | +2.050520 |

Interpretation:

- The released large-video full checkpoint is strongly target-corruption
  sensitive on both ARKit and ScanNet.
- This supports the broader Concerto-family shortcut claim.
- Main-variant claims should still use `concerto_base_origin.pth` diagnostics
  as the primary evidence.

## Key Negative Intervention Numbers

From `results_sr_lora_phasea.md`:

- SR-LoRA v5 was stable as a training intervention on large-video.
- Downstream was no-go:
  - `m=0.1`: only `+0.0009/+0.0015` ScanNet linear last/best mIoU, no stress gain.
  - `m=0.2`: `-0.0003/-0.0003` ScanNet linear last/best mIoU, no stress gain.

Interpretation:

- Large-video line shows objective-level shortcut signal.
- A global coord-rival SR-LoRA intervention is not enough to improve ScanNet
  downstream.
- This negative result should be kept, but not used as a substitute for
  origin/main-variant decoder or FT diagnostics.

## Mainline Separation Rule

When writing or deciding next experiments:

- Use origin/main-variant results as the primary line:
  - `concerto_base_origin.pth`
  - `results_main_variant_causal_battery.md`
  - `results_official_coord_mlp_rival.md`
  - `results_scannet_decoder_probe_origin.md`
  - `results_scannet_decoder_probe_origin_stagewise.md`
- Use large-video results only as:
  - cross-variant support;
  - historical negative-result evidence;
  - a reason to avoid repeating known no-go sweeps.
