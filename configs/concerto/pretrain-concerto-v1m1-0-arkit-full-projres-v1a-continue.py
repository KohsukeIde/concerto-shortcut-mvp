_base_ = ["./pretrain-concerto-v1m1-0-arkit-full-continue-a1004.py"]

_env = __import__("os").environ

model = dict(
    shortcut_probe=dict(
        mode="coord_projection_residual",
        freeze_student_backbone=False,
        zero_color=False,
        zero_normal=False,
        coord_normalize=True,
        coord_jitter_std=0.0,
        coord_prior_path=_env.get("COORD_PRIOR_PATH", ""),
        coord_projection_alpha=float(_env.get("COORD_PROJECTION_ALPHA", "0.05")),
        shuffle_correspondence=False,
    )
)

del _env
