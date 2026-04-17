_base_ = ["./pretrain-concerto-v1m1-0-probe-enc2d-baseline.py"]

_env = __import__("os").environ

num_worker = int(_env.get("CONCERTO_NUM_WORKER", "8"))
enable_wandb = False

model = dict(
    backbone_out_channels=1664,
    backbone=dict(
        enc_channels=(64, 128, 256, 512, 768),
        enc_num_head=(4, 8, 16, 32, 48),
        enable_flash=_env.get("CONCERTO_ENABLE_FLASH", "0") == "1",
    ),
    head_in_channels=1536,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    enc2d_head_in_channels=1536,
    enc2d_head_hidden_channels=4096,
    enc2d_head_embed_channels=256,
    enc2d_head_num_prototypes=4096,
    shortcut_probe=dict(
        mode="none",
        freeze_student_backbone=False,
        zero_color=False,
        zero_normal=False,
        coord_jitter_std=0.0,
        shuffle_correspondence=False,
    ),
)

del _env
