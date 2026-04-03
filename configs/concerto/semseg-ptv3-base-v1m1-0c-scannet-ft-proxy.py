_base_ = ["./semseg-ptv3-base-v1m1-0c-scannet-ft.py"]

batch_size = 16
num_worker = 16
enable_amp = True
amp_dtype = "bfloat16"
enable_wandb = False
empty_cache = False
epoch = 50
eval_epoch = 10

model = dict(
    backbone=dict(
        enable_flash=False,
    )
)
