_base_ = ["./pretrain-concerto-v1m1-0-arkit-full-continue.py"]

_env = __import__("os").environ

# H100 multi-node profile. Pointcept treats batch_size as global batch size.
# With rt_QF=8 this is 32 GPUs, so the default keeps the previous continuation
# effective batch of 32 * 3 = 96 while using one sample per GPU.
batch_size = int(_env.get("CONCERTO_GLOBAL_BATCH_SIZE", "32"))
gradient_accumulation_steps = int(_env.get("CONCERTO_GRAD_ACCUM", "3"))
num_worker = int(_env.get("CONCERTO_NUM_WORKER", "64"))
_max_train_iter = int(_env.get("CONCERTO_MAX_TRAIN_ITER", "0"))
_epoch = int(_env.get("CONCERTO_EPOCH", "0"))
_enable_flash = _env.get("CONCERTO_ENABLE_FLASH", "1") == "1"
max_train_iter_per_epoch = _max_train_iter or None
if _epoch:
    epoch = _epoch
    eval_epoch = _epoch
empty_cache = True

model = dict(
    backbone=dict(
        enable_flash=_enable_flash,
        enable_rpe=False,
        upcast_attention=False,
        upcast_softmax=False,
    )
)

del _enable_flash
del _epoch
del _max_train_iter
del _env
