import os

_base_ = ["../_base_/default_runtime.py"]

# Same-checkpoint plain LoRA per-class control for the released Concerto paper
# backbone. This follows the existing Concerto LoRA config style: PTv3 encoder
# feature (`enc_mode=True`) + LoRA on encoder qkv + linear segmentation head.

batch_size = int(os.environ.get("CONCERTO_GLOBAL_BATCH_SIZE", "64"))
num_worker = int(os.environ.get("CONCERTO_NUM_WORKER", "64"))
mix_prob = 0.8
clip_grad = 3.0
empty_cache = False
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
_enable_flash = os.environ.get("CONCERTO_ENABLE_FLASH", "1") != "0"
_enable_precise_eval = os.environ.get("CONCERTO_ENABLE_PRECISE_EVAL", "0") == "1"

del os

model = dict(
    type="DefaultLORASegmentorV2",
    num_classes=20,
    backbone_out_channels=1232,
    use_lora=True,
    lora_r=_lora_rank,
    lora_alpha=_lora_alpha,
    lora_dropout=_lora_dropout,
    keywords="module.student.backbone",
    replacements="module.backbone",
    backbone_path=_backbone_path,
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=_enable_flash,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=True,
        traceable=True,
        mask_token=False,
        freeze_encoder=True,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=False,
)

optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

dataset_type = "ScanNetDataset"
data_root = "data/scannet"

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=1.0),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1)],
                [
                    dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)

hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

if _enable_precise_eval:
    hooks.append(dict(type="PreciseEvaluator", test_last=False))
