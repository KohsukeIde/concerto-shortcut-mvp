_base_ = ["./pretrain-concerto-v1m1-0-arkit-full-continue-h10016.py"]

# Env-tunable version of the xyz-only continuation companion.
# Use CONCERTO_GLOBAL_BATCH_SIZE / CONCERTO_GRAD_ACCUM / CONCERTO_NUM_WORKER /
# CONCERTO_EPOCH / CONCERTO_ENABLE_FLASH from the parent config.
model = dict(
    shortcut_probe=dict(
        mode="none",
        freeze_student_backbone=False,
        zero_color=True,
        zero_normal=True,
        coord_jitter_std=0.0,
        shuffle_correspondence=False,
    )
)
