# Concerto Shortcut MVP Results

Setup:
- Repo: `Pointcept v1.6.1` with the shortcut probe patch in [`pointcept/models/concerto/concerto_v1m1_base.py`](../../pointcept/models/concerto/concerto_v1m1_base.py)
- Environment: `pointcept-cu128`
- Data: mini ARKitScenes subset at `data/arkitscenes`
- Budget: single GPU (`CUDA_VISIBLE_DEVICES=1`), ~7 minutes per run
- Probe config family: [`configs/concerto`](../../configs/concerto)

Notes:
- The `enc2d-only` fast path is enabled when the other SSL loss weights are zero.
- Several batches have `img_num == 0`, which yields a zero tensor loss. For comparison, the epoch-level `Train result` is more reliable than per-step minima.
- `jitter` was interrupted after 3 completed epochs because the key diagnostic runs had already finished.

Epoch-level `Train result` summary:

| Run | Completed epoch averages (`enc2d_loss`) | Last completed epoch |
| --- | --- | --- |
| `baseline` | `5.8446, 5.2548, 5.4564, 5.0278, 5.5502` | `5.5502` |
| `zero-appearance` | `5.8350, 5.4987, 5.5466, 5.7864, 4.8055` | `4.8055` |
| `coord-mlp` | `5.8603, 5.4580, 5.6214, 5.4000, 5.5692` | `5.5692` |
| `jitter` | `5.6721, 5.8570, 5.2975` | `5.2975` |
| `shuffle-corr` | `5.3948, 5.3635, 5.6940, 5.5710, 5.3146` | `5.3146` |

Takeaways:
- `coord-mlp` stays very close to `baseline`, which is strong MVP evidence for a coordinate shortcut.
- `zero-appearance` does not hurt and is actually better on this mini setup, suggesting weak dependence on color/normal appearance.
- `jitter` also stays close to `baseline`, consistent with weak dependence on local geometry.
- `shuffle-corr` does **not** degrade relative to `baseline` in this MVP. That means the current shuffle probe is not a positive sanity check. The next iteration should use a stronger correspondence corruption (for example cross-scene or cross-image reassignment).
