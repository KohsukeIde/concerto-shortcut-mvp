# ScanNet Gate Result (2026-04-09)

## Bottom Line

The safe single-GPU ScanNet linear gate is now confirmed to run end-to-end.
This means the downstream pipeline is alive on the current machine under the
safe proxy setup.

What this does **not** establish yet:
- It does not answer the downstream replacement question.
- It does not yet tell us whether `coord_mlp` preserves the Concerto gain.

That go/no-go remains pending until the continuation trio and ScanNet linear
trio finish.

## Gate Result

Source log:
- [exp/concerto/scannet-proxy-official-origin-lin/train.log](../../exp/concerto/scannet-proxy-official-origin-lin/train.log)

Final validation result:
- `mIoU = 0.1752`
- `mAcc = 0.2467`
- `allAcc = 0.6167`

Checkpoint artifacts:
- [exp/concerto/scannet-proxy-official-origin-lin/model/model_best.pth](../../exp/concerto/scannet-proxy-official-origin-lin/model/model_best.pth)
- [exp/concerto/scannet-proxy-official-origin-lin/model/model_last.pth](../../exp/concerto/scannet-proxy-official-origin-lin/model/model_last.pth)

## Interpretation

- The original 2-GPU DDP gate was unstable in the `mp.spawn` / NCCL path.
- The safe single-GPU path works.
- Therefore the downstream blocker is no longer "can ScanNet run at all?".
- The remaining blocker is simply finishing the replacement comparison jobs.

## Current Follow-up Jobs

Running:
- `no-enc2d` continuation
  - [exp/concerto/arkit-full-continue-no-enc2d/train.log](../../exp/concerto/arkit-full-continue-no-enc2d/train.log)
- `coord_mlp` continuation
  - [exp/concerto/arkit-full-continue-coord-mlp-debug2/train.log](../../exp/concerto/arkit-full-continue-coord-mlp-debug2/train.log)

Queued / conditional:
- `concerto_continue`
- `ScanNet linear trio` after the three continuation checkpoints exist

Follow-up chain:
- service: `concerto-safe-followup`
- log: [logs/safe_followup_chain.log](./logs/safe_followup_chain.log)

## Current Decision

- Objective-level shortcut claim: `go`
- Downstream pipeline viability: `go`
- Downstream main-claim decision: `pending`
