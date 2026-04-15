#!/usr/bin/env python3
"""Small torchrun diagnostic for ARKit DDP batch loading on ABCI-Q."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_config(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path.resolve()
    candidate = repo_root / "configs" / "concerto" / raw
    if candidate.exists():
        return candidate.resolve()
    candidate = repo_root / "configs" / "concerto" / f"{raw}.py"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(raw)


def setup_local_process_group(local_world_size: int, node_rank: int) -> None:
    import pointcept.utils.comm as comm

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


def recursive_cuda(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.cuda(non_blocking=True)
    if isinstance(value, dict):
        return {key: recursive_cuda(item) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_cuda(item) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_cuda(item) for item in value)
    return value


def summarize_value(value: Any) -> str:
    if torch.is_tensor(value):
        return f"tensor{tuple(value.shape)}:{value.dtype}"
    if isinstance(value, (list, tuple)):
        preview = ",".join(summarize_value(item) for item in list(value)[:3])
        suffix = ",..." if len(value) > 3 else ""
        return f"{type(value).__name__}[{len(value)}]({preview}{suffix})"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    return repr(value)[:80]


def summarize_batch(batch: dict[str, Any]) -> str:
    keys = sorted(batch.keys())
    focus = []
    for key in ("name", "img_num", "coord", "images", "global_correspondence"):
        if key in batch:
            focus.append(f"{key}={summarize_value(batch[key])}")
    return f"keys={keys}; " + "; ".join(focus)


def build_train_loader(cfg):
    import pointcept.utils.comm as comm
    from pointcept.datasets import build_dataset, point_collate_fn
    from pointcept.engines.defaults import worker_init_fn

    train_data = build_dataset(cfg.data.train)
    if comm.get_world_size() > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        sampler = None

    init_fn = (
        partial(
            worker_init_fn,
            num_workers=cfg.num_worker_per_gpu,
            rank=comm.get_rank(),
            seed=cfg.seed,
        )
        if cfg.seed is not None
        else None
    )
    loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.batch_size_per_gpu,
        shuffle=(sampler is None),
        num_workers=cfg.num_worker_per_gpu,
        sampler=sampler,
        collate_fn=partial(point_collate_fn, mix_prob=cfg.mix_prob),
        pin_memory=getattr(cfg, "pin_memory", True),
        worker_init_fn=init_fn,
        drop_last=len(train_data) > cfg.batch_size,
        persistent_workers=(
            cfg.num_worker_per_gpu > 0 and getattr(cfg, "persistent_workers", True)
        ),
    )
    return train_data, loader


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug ARKit DDP batch loading without running the model."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument(
        "--config",
        default="pretrain-concerto-v1m1-0-arkit-full-projres-v1a-smoke-h10016",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-worker", type=int, default=None)
    parser.add_argument("--dist-timeout-sec", type=int, default=180)
    parser.add_argument("--no-cuda-copy", action="store_true")
    parser.add_argument("--no-allreduce", action="store_true")
    parser.add_argument("--barrier", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    node_rank = int(os.environ.get("GROUP_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=args.dist_timeout_sec),
        )
        setup_local_process_group(local_world_size, node_rank)

    from pointcept.engines.defaults import default_setup
    from pointcept.utils.config import Config

    config_path = resolve_config(repo_root, args.config)
    cfg = Config.fromfile(str(config_path))
    default_save_path = repo_root / "data" / "runs" / "projres_v1" / "debug" / "ddp_batches"
    cfg.save_path = str((args.save_path or default_save_path).resolve())
    cfg.resume = False
    cfg.enable_wandb = False
    if args.data_root is not None:
        cfg.data.train.data_root = str(args.data_root.resolve())
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_worker is not None:
        cfg.num_worker = args.num_worker

    cfg = default_setup(cfg)
    if rank == 0:
        Path(cfg.save_path).mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    dataset, loader = build_train_loader(cfg)
    if getattr(loader, "sampler", None) is not None and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(0)

    print(
        "[ddp-batch] "
        f"rank={rank}/{world_size} local_rank={local_rank} "
        f"dataset_len={len(dataset)} loader_len={len(loader)} "
        f"batch_size_per_gpu={cfg.batch_size_per_gpu} "
        f"num_worker_per_gpu={cfg.num_worker_per_gpu} "
        f"cuda_copy={not args.no_cuda_copy} allreduce={not args.no_allreduce}",
        flush=True,
    )

    iterator = iter(loader)
    for step in range(args.max_batches):
        before = time.monotonic()
        print(f"[ddp-batch] rank={rank} step={step} before_next", flush=True)
        try:
            batch = next(iterator)
        except StopIteration:
            print(f"[ddp-batch] rank={rank} step={step} stop_iteration", flush=True)
            break
        after_next = time.monotonic()
        print(
            f"[ddp-batch] rank={rank} step={step} after_next "
            f"dt={after_next - before:.3f}s {summarize_batch(batch)}",
            flush=True,
        )

        if not args.no_cuda_copy:
            batch = recursive_cuda(batch)
            torch.cuda.synchronize()
            print(
                f"[ddp-batch] rank={rank} step={step} after_cuda "
                f"dt={time.monotonic() - after_next:.3f}s",
                flush=True,
            )

        if dist.is_initialized() and not args.no_allreduce:
            token = torch.tensor([float(step + 1)], device="cuda")
            dist.all_reduce(token)
            torch.cuda.synchronize()
            print(
                f"[ddp-batch] rank={rank} step={step} after_allreduce "
                f"value={token.item():.1f}",
                flush=True,
            )

        if dist.is_initialized() and args.barrier:
            dist.barrier()
            print(f"[ddp-batch] rank={rank} step={step} after_barrier", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    print(f"[ddp-batch] rank={rank} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
