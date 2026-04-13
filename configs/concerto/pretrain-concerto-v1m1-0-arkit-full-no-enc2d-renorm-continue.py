_base_ = ["./pretrain-concerto-v1m1-0-arkit-full-continue.py"]

model = dict(
    mask_loss_weight=1 / 4,
    roll_mask_loss_weight=1 / 4,
    unmask_loss_weight=1 / 2,
    enc2d_loss_weight=0.0,
)
