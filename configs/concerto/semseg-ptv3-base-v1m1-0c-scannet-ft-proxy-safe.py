_base_ = ["./semseg-ptv3-base-v1m1-0c-scannet-ft-proxy.py"]

batch_size = 8
num_worker = 4
enable_amp = False
pin_memory = False
persistent_workers = False
