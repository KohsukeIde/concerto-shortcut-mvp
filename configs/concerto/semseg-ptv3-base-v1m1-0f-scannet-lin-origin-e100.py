_base_ = ["./semseg-ptv3-base-v1m1-0f-scannet-ft-lora-origin-e100.py"]

# Same-head baseline for the origin LoRA control:
# frozen Concerto origin encoder features + linear segmentation head only.
# This keeps the data, schedule, optimizer, and head family identical to the
# plain LoRA control while disabling LoRA adapters.

model = dict(use_lora=False)
