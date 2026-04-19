# Official Concerto LoRA Recipe Check

## Summary

The upstream Pointcept Concerto LoRA recipe is **materially different** from the
decoder-capacity-matched LoRA control run here.

The upstream recipe is an encoder-mode large-model LoRA setup:

- config: `configs/concerto/semseg-ptv3-large-v1m1-0f-scannet-ft-lora.py`
- model: `DefaultLORASegmentorV2`
- backbone size: PTv3 **large**
- `backbone_out_channels=1728`
- `enc_channels=(64, 128, 256, 512, 768)`
- `enc_mode=True`
- no PTv3 decoder branch
- segmentation head: linear head on the pooled encoder-mode feature
- LoRA: qkv adapters, rank `8`, alpha `16`, dropout `0.1`
- trainable parameters: segmentation head plus qkv LoRA adapters
- schedule: `800` epochs, batch size `24`, AdamW `lr=0.002`,
  block/LoRA group lr `0.0002`
- checkpoint expected by config:
  `exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth`

The local decoder-capacity-matched control is not this recipe:

- config: `configs/concerto/semseg-ptv3-base-v1m1-0c-scannet-dec-lora-origin-e100.py`
- backbone size: PTv3 **base**
- checkpoint: `data/weights/concerto/concerto_base_origin.pth`
- `backbone_out_channels=64`
- `enc_mode=False`
- PTv3 decoder branch is present and trainable
- schedule: `100` epochs, batch size `64`, AdamW `lr=0.001`,
  block/LoRA group lr `0.0001`

Therefore the completed decoder+LoRA no-go should **not** be treated as an
official Concerto LoRA reproduction. It answers only the matched decoder-family
question for the released paper-origin base checkpoint.

## Sources Checked

- Local repo config:
  `configs/concerto/semseg-ptv3-large-v1m1-0f-scannet-ft-lora.py`
- Upstream Pointcept config:
  <https://raw.githubusercontent.com/Pointcept/Pointcept/main/configs/concerto/semseg-ptv3-large-v1m1-0f-scannet-ft-lora.py>
- Local implementation:
  `pointcept/models/default.py`
- Upstream Pointcept implementation:
  <https://raw.githubusercontent.com/Pointcept/Pointcept/main/pointcept/models/default.py>

## Trainable Scope

`DefaultLORASegmentorV2` loads the pretrained backbone, injects PEFT LoRA into
`self.backbone.enc` with `target_modules=["qkv"]`, then leaves LoRA parameters
and the segmentation head trainable. In the PTv3 config, `freeze_encoder=True`
freezes the embedding and encoder before LoRA is injected; LoRA params are then
explicitly re-enabled.

One implementation detail differs between this local repo and upstream
Pointcept: the local `get_peft_model` call uses `autocast_adapter_dtype=False`,
while the upstream main branch call does not. This is worth preserving in notes
for exact reproduction, but it is secondary to the much larger config-level
difference between encoder-mode large LoRA and local decoder+LoRA.

So the upstream recipe is not "decoder+LoRA". It is closer to the completed
same-head linear LoRA control, but with important differences:

| item | local same-head origin LoRA | upstream Concerto LoRA |
| --- | --- | --- |
| checkpoint | `concerto_base_origin.pth` | `pretrain-concerto-v1m1-0-large-base/model/model_last.pth` |
| model size | base | large |
| `backbone_out_channels` | 1232 | 1728 |
| head family | encoder-mode linear head | encoder-mode linear head |
| `enc_mode` | `True` | `True` |
| decoder branch | none | none |
| epochs used here / upstream | 100 | 800 |
| batch size | 64 | 24 |
| AdamW lr | 0.001 | 0.002 |
| block/LoRA lr | 0.0001 | 0.0002 |

## Local Checkpoint Availability

Currently present under `data/weights/concerto`:

- `concerto_base_origin.pth`
- `concerto_base.pth`
- `pretrain-concerto-v1m1-2-large-video.pth`

The exact upstream config checkpoint,
`exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth`, is not
present locally at the checked path.

## Decision

Do not interpret the completed decoder+LoRA no-go as an official LoRA no-go.

If we want to answer the official LoRA question, the next required step is to
obtain the main-variant large pretraining checkpoint expected by the upstream
config, or construct a carefully labeled substitute. Without that checkpoint,
the closest local result is the base-origin same-head LoRA control, not the
official large LoRA recipe.
