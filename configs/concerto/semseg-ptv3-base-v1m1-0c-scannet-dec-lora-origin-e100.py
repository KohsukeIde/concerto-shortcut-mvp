import os

_base_ = ["./semseg-ptv3-base-v1m1-0c-scannet-dec.py"]

# Decoder-capacity-matched LoRA control for the released Concerto paper
# backbone. This keeps the decoder-probe family (`backbone_out_channels=64`,
# `enc_mode=False`, trainable decoder) and adds LoRA only to encoder qkv.
# Use this to compare against `semseg-ptv3-base-v1m1-0c-scannet-dec-origin-e100`
# without the linear-head capacity confound.

batch_size = int(os.environ.get("CONCERTO_GLOBAL_BATCH_SIZE", "64"))
num_worker = int(os.environ.get("CONCERTO_NUM_WORKER", "64"))
enable_amp = os.environ.get("CONCERTO_ENABLE_AMP", "1") != "0"

epoch = int(os.environ.get("CONCERTO_EPOCH", "100"))
eval_epoch = int(os.environ.get("CONCERTO_EVAL_EPOCH", str(epoch)))

_lora_rank = int(os.environ.get("SCANNET_LORA_RANK", "8"))
_lora_alpha = int(os.environ.get("SCANNET_LORA_ALPHA", str(_lora_rank * 2)))
_lora_dropout = float(os.environ.get("SCANNET_LORA_DROPOUT", "0.1"))
_backbone_path = os.environ.get(
    "LORA_BACKBONE_PATH",
    "data/weights/concerto/concerto_base_origin.pth",
)
_enable_precise_eval = os.environ.get("CONCERTO_ENABLE_PRECISE_EVAL", "0") == "1"

model = dict(
    type="DefaultLORASegmentorV2",
    use_lora=True,
    lora_r=_lora_rank,
    lora_alpha=_lora_alpha,
    lora_dropout=_lora_dropout,
    keywords="module.student.backbone",
    replacements="module.backbone",
    backbone_path=_backbone_path,
)

# The model self-loads the frozen origin backbone. Passing WEIGHT_PATH=None keeps
# the training hook from loading a second checkpoint over the LoRA-wrapped model.
hooks = [
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

if _enable_precise_eval:
    hooks.append(dict(type="PreciseEvaluator", test_last=False))

del os
