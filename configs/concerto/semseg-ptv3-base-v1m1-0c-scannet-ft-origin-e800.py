_base_ = ["./semseg-ptv3-base-v1m1-0c-scannet-ft.py"]

# Official-like full-FT control for the released Concerto paper backbone.
# Keep the upstream ScanNet FT recipe unchanged except for loading the released
# backbone checkpoint with the resolved prefix mapping.
epoch = 800
eval_epoch = 100

hooks = [
    dict(type="CheckpointLoader", keywords="module", replacement="module.backbone"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
