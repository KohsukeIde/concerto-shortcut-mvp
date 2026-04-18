_base_ = ["./semseg-ptv3-base-v1m1-0c-scannet-dec.py"]

# Smoke config for the released Concerto paper backbone
# `data/weights/concerto/concerto_base_origin.pth`.
epoch = 1
eval_epoch = 1

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module.backbone."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
