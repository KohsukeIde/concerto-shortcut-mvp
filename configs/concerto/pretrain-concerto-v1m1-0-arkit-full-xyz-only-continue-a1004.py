_base_ = ["./pretrain-concerto-v1m1-0-arkit-full-continue-a1004.py"]

# Representation-level coordinate-input companion:
# keep the original cross-modal target and Concerto/PTv3 capacity, but restrict
# the point-side input features to xyz by zeroing appearance and normal channels.
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
