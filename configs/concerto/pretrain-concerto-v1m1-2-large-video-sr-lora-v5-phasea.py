import os
import runpy
from pathlib import Path

_repo_root = Path(os.environ.get("REPO_ROOT", Path.cwd()))
_base_values = runpy.run_path(
    str(_repo_root / "configs" / "concerto" / "pretrain-concerto-v1m1-2-large-video.py")
)
for _key, _value in _base_values.items():
    if not _key.startswith("__"):
        globals()[_key] = _value

_env = os.environ


def _pick_root(*candidates, default):
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)
    return default


batch_size = int(_env.get("CONCERTO_GLOBAL_BATCH_SIZE", "48"))
num_worker = int(_env.get("CONCERTO_NUM_WORKER", "64"))
epoch = int(_env.get("CONCERTO_EPOCH", "5"))
eval_epoch = int(_env.get("CONCERTO_EVAL_EPOCH", str(epoch)))
stop_epoch = int(_env.get("CONCERTO_STOP_EPOCH", str(epoch)))
_max_train_iter = int(_env.get("CONCERTO_MAX_TRAIN_ITER", "0"))
max_train_iter_per_epoch = _max_train_iter or None
base_lr = float(_env.get("SR_BASE_LR", "0.0002"))
base_wd = float(_env.get("SR_BASE_WD", "0.04"))
final_wd = float(_env.get("SR_FINAL_WD", "0.2"))
enable_amp = True
amp_dtype = "bfloat16"
enable_wandb = False
find_unused_parameters = True

_sr_rank = int(_env.get("SR_LORA_RANK", "4"))
_sr_alpha = int(_env.get("SR_LORA_ALPHA", str(_sr_rank * 2)))
_coord_rival_path = _env.get("COORD_RIVAL_PATH")
if not _coord_rival_path:
    raise RuntimeError("COORD_RIVAL_PATH is required for SR-LoRA Phase A.")

model["backbone"]["enable_flash"] = _env.get("CONCERTO_ENABLE_FLASH", "1") == "1"
model.update(
    momentum_base=1.0,
    momentum_final=1.0,
    shortcut_probe=dict(
        mode="coord_margin_rival",
        freeze_student_backbone=False,
        zero_color=False,
        zero_normal=False,
        coord_jitter_std=0.0,
        shuffle_correspondence=False,
        coord_normalize=True,
        coord_probe_hidden_channels=int(_env.get("COORD_RIVAL_HIDDEN", "512")),
        coord_rival_path=_coord_rival_path,
        sr_margin_alpha=float(_env.get("SR_MARGIN_ALPHA", "1.0")),
        sr_margin_value=float(_env.get("SR_MARGIN_VALUE", "0.1")),
        sr_distill_weight=float(_env.get("SR_DISTILL_WEIGHT", "0.3")),
        sr_lora_enable=True,
        sr_lora_r=_sr_rank,
        sr_lora_alpha=_sr_alpha,
        sr_lora_dropout=float(_env.get("SR_LORA_DROPOUT", "0.05")),
        sr_train_patch_proj=_env.get("SR_TRAIN_PATCH_PROJ", "0") == "1",
        sr_train_student_heads=_env.get("SR_TRAIN_STUDENT_HEADS", "0") == "1",
    ),
)

dec_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1),
    )
    for e in range(len(dec_depths))
    for b in range(dec_depths[e])
]
del dec_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [group["lr"] for group in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

_max_size = int(_env.get("CONCERTO_MAX_SIZE", "65536"))
_enc2d_max_size = int(_env.get("CONCERTO_ENC2D_MAX_SIZE", str(_max_size)))
for _item in transform:
    if _item.get("type") == "MultiViewGenerator":
        _item["max_size"] = _max_size
        _item["enc2d_max_size"] = _enc2d_max_size

arkit_root = _pick_root(
    _env.get("ARKIT_FULL_META_ROOT"),
    _env.get("ARKIT_FULL_SOURCE_ROOT"),
    default="data/arkitscenes_absmeta",
)
scannet_root = _pick_root(
    _env.get("SCANNET_IMAGEPOINT_META_ROOT"),
    _env.get("SCANNET_IMAGEPOINT_ROOT"),
    default="data/concerto_scannet_imagepoint_absmeta",
)
scannetpp_root = _pick_root(
    _env.get("SCANNETPP_IMAGEPOINT_META_ROOT"),
    _env.get("SCANNETPP_IMAGEPOINT_ROOT"),
    default="data/concerto_scannetpp_imagepoint_absmeta",
)
s3dis_root = _pick_root(
    _env.get("S3DIS_IMAGEPOINT_META_ROOT"),
    _env.get("S3DIS_IMAGEPOINT_ROOT"),
    default="data/concerto_s3dis_imagepoint_absmeta",
)
hm3d_root = _pick_root(
    _env.get("HM3D_IMAGEPOINT_META_ROOT"),
    _env.get("HM3D_IMAGEPOINT_ROOT"),
    default="data/concerto_hm3d_imagepoint_absmeta",
)
structured3d_root = _pick_root(
    _env.get("STRUCTURED3D_IMAGEPOINT_META_ROOT"),
    _env.get("STRUCTURED3D_IMAGEPOINT_ROOT"),
    default="data/concerto_structured3d_imagepoint_absmeta",
)


def _loop_for(name):
    return int(_env.get(f"{name.upper()}_LOOP", _env.get("CONCERTO_DATASET_LOOP", "1")))


data = dict(
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(type="DefaultImagePointDataset", crop_h=crop_h, crop_w=crop_w, patch_size=patch_size, split=["Training", "Validation"], data_root=arkit_root, transform=transform, test_mode=False, loop=_loop_for("arkit")),
            dict(type="DefaultImagePointDataset", crop_h=crop_h, crop_w=crop_w, patch_size=patch_size, split=["train", "val", "test"], data_root=scannet_root, transform=transform, test_mode=False, loop=_loop_for("scannet")),
            dict(type="DefaultImagePointDataset", crop_h=crop_h, crop_w=crop_w, patch_size=patch_size, split=["train", "val", "test"], data_root=scannetpp_root, transform=transform, test_mode=False, loop=_loop_for("scannetpp")),
            dict(type="DefaultImagePointDataset", crop_h=crop_h, crop_w=crop_w, patch_size=patch_size, split=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"], data_root=s3dis_root, transform=transform, test_mode=False, loop=_loop_for("s3dis")),
            dict(type="DefaultImagePointDataset", crop_h=crop_h, crop_w=crop_w, patch_size=patch_size, split=["train", "val"], data_root=hm3d_root, transform=transform, test_mode=False, loop=_loop_for("hm3d")),
            dict(type="DefaultImagePointDataset", crop_h=crop_h, crop_w=crop_w, patch_size=patch_size, split=["train", "val"], data_root=structured3d_root, transform=transform, test_mode=False, loop=_loop_for("structured3d")),
        ],
    )
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=max(1, epoch)),
]

del _env, _repo_root, _base_values, _key, _value, _max_train_iter
del _pick_root, _loop_for, _item, os, runpy, Path
