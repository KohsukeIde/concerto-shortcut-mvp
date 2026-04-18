_base_ = ["./semseg-ptv3-base-v1m1-0c-scannet-dec.py"]

# 100-epoch decoder-probe diagnostic for the released Concerto paper backbone.
# Keeps the official base recipe batch_size=64 and lr=0.001 from the base config.
epoch = 100
eval_epoch = 100

hooks = [
    dict(type="CheckpointLoader", keywords="module", replacement="module.backbone"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
