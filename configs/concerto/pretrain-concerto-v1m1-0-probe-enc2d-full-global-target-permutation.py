_base_ = ["./pretrain-concerto-v1m1-0-probe-enc2d-full-baseline.py"]

model = dict(
    shortcut_probe=dict(
        mode="global_target_permutation",
        freeze_student_backbone=False,
        zero_color=False,
        zero_normal=False,
        coord_jitter_std=0.0,
        shuffle_correspondence=False,
    )
)
