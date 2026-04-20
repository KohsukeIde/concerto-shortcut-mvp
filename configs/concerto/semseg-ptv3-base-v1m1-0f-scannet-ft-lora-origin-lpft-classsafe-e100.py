import os

_base_ = ["./semseg-ptv3-base-v1m1-0f-scannet-ft-lora-origin-e100.py"]

# LP-FT3D / class-safe LoRA follow-up.
#
# This keeps the official-like linear-head family where plain LoRA was a
# matched-head positive control. The job should be launched with WEIGHT_PATH
# pointing to the trained no-LoRA linear checkpoint so that the segmentation
# head is warm-started. A frozen copy of that same checkpoint is used as the
# non-weak-class safety anchor.

_anchor_path = os.environ.get(
    "CLASS_SAFE_ANCHOR_PATH",
    "exp/concerto/scannet-lin-origin-e100/model/model_best.pth",
)
_weak_loss_weight = float(os.environ.get("CLASS_SAFE_WEAK_LOSS_WEIGHT", "0.2"))
_safe_kl_weight = float(os.environ.get("CLASS_SAFE_KL_WEIGHT", "0.05"))
_dist_kl_weight = float(os.environ.get("CLASS_SAFE_DIST_KL_WEIGHT", "0.02"))
_kl_temperature = float(os.environ.get("CLASS_SAFE_KL_TEMPERATURE", "2.0"))

model = dict(
    type="DefaultClassSafeLORASegmentorV2",
    anchor_path=_anchor_path,
    anchor_keywords="module",
    anchor_replacements="module",
    weak_classes=(10, 11, 17, 15, 7),
    weak_loss_weight=_weak_loss_weight,
    safe_kl_weight=_safe_kl_weight,
    dist_kl_weight=_dist_kl_weight,
    kl_temperature=_kl_temperature,
)

del os
