_base_ = ["./semseg-ptv3-base-v1m1-0a-scannet-lin-proxy-valonly.py"]

_env = __import__("os").environ

model = dict(
    type="PosthocEditedSegmentorV2",
    freeze_backbone=True,
    backbone_path=_env.get("POSTHOC_BACKBONE_CKPT", ""),
    keywords=_env.get("POSTHOC_BACKBONE_KEYWORDS", "module.student.backbone"),
    replacements=_env.get("POSTHOC_BACKBONE_REPLACEMENTS", "module.backbone"),
    editor_cfg=dict(
        type="Splice3DEditor",
        feature_dim=1232,
        preserve_mean=True,
    ),
    editor_path=_env.get("POSTHOC_EDITOR_CKPT", ""),
    return_edit_metrics=True,
)

hooks = [
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

del _env
