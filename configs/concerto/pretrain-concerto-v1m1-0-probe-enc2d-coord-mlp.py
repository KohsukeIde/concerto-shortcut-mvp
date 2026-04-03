_base_ = ["./pretrain-concerto-v1m1-0-probe-enc2d-baseline.py"]

model = dict(
    shortcut_probe=dict(
        mode="coord_mlp",
        freeze_student_backbone=True,
        zero_color=True,
        zero_normal=True,
        coord_normalize=True,
        coord_jitter_std=0.0,
        shuffle_correspondence=False,
    )
)
