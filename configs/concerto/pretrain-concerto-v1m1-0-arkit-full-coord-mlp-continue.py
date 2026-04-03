_base_ = ["./pretrain-concerto-v1m1-0-arkit-full-continue.py"]

model = dict(
    shortcut_probe=dict(
        mode="coord_mlp",
        freeze_student_backbone=False,
        zero_color=False,
        zero_normal=False,
        coord_normalize=True,
        coord_jitter_std=0.0,
        shuffle_correspondence=False,
    )
)
