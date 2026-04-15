_base_ = ["./pretrain-concerto-v1m1-0-arkit-full-projres-v1a-continue.py"]

_env = __import__("os").environ

# H100 x16 profile. Pointcept interprets batch_size as global batch size,
# so the default below is 2 samples/GPU with grad accumulation to match the
# original Concerto effective global batch of 96.
batch_size = int(_env.get("CONCERTO_GLOBAL_BATCH_SIZE", "32"))
gradient_accumulation_steps = int(_env.get("CONCERTO_GRAD_ACCUM", "3"))
num_worker = int(_env.get("CONCERTO_NUM_WORKER", "64"))
_max_train_iter = int(_env.get("CONCERTO_MAX_TRAIN_ITER", "0"))
max_train_iter_per_epoch = _max_train_iter or None
empty_cache = True

model = dict(
    backbone=dict(
        enable_flash=True,
        enable_rpe=False,
        upcast_attention=False,
        upcast_softmax=False,
    )
)

del _max_train_iter
del _env
