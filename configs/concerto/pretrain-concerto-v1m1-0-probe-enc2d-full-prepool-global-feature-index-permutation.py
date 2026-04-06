_base_ = ["./pretrain-concerto-v1m1-0-probe-enc2d-full-baseline.py"]

model = dict(
    shortcut_probe=dict(
        mode="prepool_global_feature_index_permutation",
    )
)
