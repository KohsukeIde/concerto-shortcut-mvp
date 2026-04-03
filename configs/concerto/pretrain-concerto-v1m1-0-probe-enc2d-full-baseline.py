_base_ = ["./pretrain-concerto-v1m1-0-probe-enc2d-baseline.py"]

num_worker = 8
enable_wandb = False

arkit_full_data_root = "/home/cvrt/datasets/arkitscenes/arkitscenes_absmeta"

data = dict(
    train=dict(
        data_root=arkit_full_data_root,
    )
)
