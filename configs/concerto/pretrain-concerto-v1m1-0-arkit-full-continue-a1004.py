_base_ = ["./pretrain-concerto-v1m1-0-arkit-full-continue.py"]

# A100 80GB x4 profile:
# keep effective batch size 8 while lowering per-rank memory pressure.
batch_size = 2
gradient_accumulation_steps = 4
num_worker = 2
empty_cache = True
