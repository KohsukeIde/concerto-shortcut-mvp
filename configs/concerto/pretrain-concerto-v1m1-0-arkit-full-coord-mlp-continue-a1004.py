_base_ = ["./pretrain-concerto-v1m1-0-arkit-full-coord-mlp-continue.py"]

batch_size = 2
gradient_accumulation_steps = 4
num_worker = 2
empty_cache = True
