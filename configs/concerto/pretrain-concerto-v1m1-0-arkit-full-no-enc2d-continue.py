_base_ = ["./pretrain-concerto-v1m1-0-arkit-full-continue.py"]

model = dict(
    enc2d_loss_weight=0.0,
)
