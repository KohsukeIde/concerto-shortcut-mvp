"""
Training entry point for torchrun / torch.distributed.run.

This mirrors tools/train.py but relies on torchrun-provided environment
variables instead of pointcept.engines.launch mp.spawn.
"""

import os

import torch
import torch.distributed as dist

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.utils import comm


def _setup_local_process_group(local_world_size: int, node_rank: int) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if local_world_size <= 0 or world_size % local_world_size != 0:
        return

    num_nodes = world_size // local_world_size
    for idx in range(num_nodes):
        ranks = list(range(idx * local_world_size, (idx + 1) * local_world_size))
        group = dist.new_group(ranks)
        if idx == node_rank:
            comm.set_local_process_group(group)


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(args.num_gpus)))
    node_rank = int(os.environ.get("GROUP_RANK", str(args.machine_rank)))

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only "
                f"{torch.cuda.device_count()} CUDA devices are visible"
            )
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        _setup_local_process_group(local_world_size, node_rank)

    if rank == 0:
        print(
            "[INFO] torchrun entry: "
            f"rank={rank} local_rank={local_rank} "
            f"world_size={world_size} local_world_size={local_world_size}",
            flush=True,
        )

    main_worker(cfg)


if __name__ == "__main__":
    main()
