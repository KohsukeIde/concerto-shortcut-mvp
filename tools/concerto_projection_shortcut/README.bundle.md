# Concerto projection/correspondence shortcut MVP

This bundle is a **minimal, code-level probe** for the Pointcept implementation of Concerto.
It is designed to answer one concrete question:

> Does the 2D-3D joint embedding loss really need rich 3D geometry, or can it be driven mostly by
> explicit point-pixel correspondence plus coarse / absolute coordinates?

## Important correction after reading the repo

The training-time `forward()` path in Concerto does **not** ingest camera intrinsics or poses directly.
The relevant side-channel in the released code is:

1. `DefaultImagePointDataset` loads precomputed `correspondence` for each point.
2. `MultiViewGenerator` keeps only image views that match the selected 3D crop.
3. The pretrain config collects `images`, `global_correspondence`, and `img_num`.
4. `concerto_v1m1_base.py` pools point features, reads `correspondence`, extracts 2D teacher patch
   features with DINOv2, and applies cosine loss between the projected 3D features and the indexed
   2D patch features.

So the shortest suspicious path is not literally “the model learns the camera matrix inside forward”.
It is closer to:

- **correspondence shortcut**: explicit point-pixel matches are strong enough that the 3D branch can
  behave like a cache for teacher patches;
- **coordinate shortcut**: a shallow function of pooled XYZ may already predict a large fraction of the
  DINO patch feature signal;
- **scene/layout shortcut**: the model may rely more on coarse spatial layout than on local geometry.

This MVP is built around those three failure modes.

## What is included

### Model patch

`overlay/pointcept/models/concerto/concerto_v1m1_base.py`

Added `shortcut_probe` options:

- `mode="none" | "coord_mlp"`
- `freeze_student_backbone`
- `zero_color`
- `zero_normal`
- `coord_jitter_std`
- `coord_jitter_clip`
- `coord_normalize`
- `shuffle_correspondence`

What each one does:

- `coord_mlp`:
  replaces the 3D feature used by the cross-modal loss with a tiny MLP over pooled XYZ only.
  This is the cleanest coordinate-only baseline.
- `zero_color` / `zero_normal`:
  removes appearance and normal cues from the student input, while keeping the same correspondence.
- `coord_jitter_std`:
  perturbs coordinates before the student backbone, while keeping the same correspondence.
  This weakens local geometry but preserves the same 2D target.
- `shuffle_correspondence`:
  shuffles correspondence within each scene. This is a sanity check that the explicit
  point-pixel side-channel is actually load-bearing.

### Full probe configs

All configs are complete copies of the upstream pretrain config with small GPU-friendly changes
(batch size 8, 5 epochs, AMP off) and one probe switch.

- `overlay/configs/concerto/pretrain-concerto-v1m1-0-probe-enc2d-baseline.py`
- `overlay/configs/concerto/pretrain-concerto-v1m1-0-probe-enc2d-zero-appearance.py`
- `overlay/configs/concerto/pretrain-concerto-v1m1-0-probe-enc2d-coord-mlp.py`
- `overlay/configs/concerto/pretrain-concerto-v1m1-0-probe-enc2d-jitter.py`
- `overlay/configs/concerto/pretrain-concerto-v1m1-0-probe-enc2d-shuffle-corr.py`

All of them isolate **only the enc2d loss** by setting:

- `mask_loss_weight = 0.0`
- `roll_mask_loss_weight = 0.0`
- `unmask_loss_weight = 0.0`
- `enc2d_loss_weight = 1.0`

That makes the probe much cleaner than mixing in the Sonata-style self-distillation losses.

### Helper scripts

- `apply_overlay.sh`
- `overlay/tools/concerto_projection_shortcut/run_mvp.sh`
- `overlay/tools/concerto_projection_shortcut/summarize_logs.py`

## How to apply

From the extracted bundle directory:

```bash
bash apply_overlay.sh /path/to/Pointcept
```

Then go to the Pointcept repo root.

## MVP experiment order

### 1. Baseline enc2d-only

```bash
bash scripts/train.sh -m 1 -g 1 -d concerto \
  -c pretrain-concerto-v1m1-0-probe-enc2d-baseline \
  -n baseline-enc2d
```

Purpose:
Get a small-resource reference curve for `enc2d_loss` when the cross-modal branch is intact.

### 2. Zero appearance + normal

```bash
bash scripts/train.sh -m 1 -g 1 -d concerto \
  -c pretrain-concerto-v1m1-0-probe-enc2d-zero-appearance \
  -n zero-appearance-enc2d
```

Interpretation:
If `enc2d_loss` stays close to baseline, the branch does not need color or normals much.
That makes a coordinate / layout shortcut much more plausible.

### 3. Coordinate-only MLP

```bash
bash scripts/train.sh -m 1 -g 1 -d concerto \
  -c pretrain-concerto-v1m1-0-probe-enc2d-coord-mlp \
  -n coord-mlp-enc2d
```

Interpretation:
This is the strongest MVP test.
If a tiny MLP over pooled XYZ alone drives `enc2d_loss` down to a non-trivial level, then a large
fraction of the released cross-modal signal is available through coordinates + correspondence alone.

### 4. Coordinate jitter

```bash
bash scripts/train.sh -m 1 -g 1 -d concerto \
  -c pretrain-concerto-v1m1-0-probe-enc2d-jitter \
  -n jitter-enc2d
```

Interpretation:
If moderate coordinate perturbation does **not** hurt much, the branch is insensitive to local 3D shape.
If it hurts sharply, the branch depends more on precise geometry.

### 5. Shuffle correspondence

```bash
bash scripts/train.sh -m 1 -g 1 -d concerto \
  -c pretrain-concerto-v1m1-0-probe-enc2d-shuffle-corr \
  -n shuffle-corr-enc2d
```

Interpretation:
This should be the destructive sanity check.
If the loss collapses here, it confirms that the explicit point-pixel mapping is a crucial side-channel.

## What result pattern would be strongest?

The strongest “shortcut” pattern is:

1. baseline is good;
2. zero-appearance remains close to baseline;
3. coord-MLP is surprisingly competitive;
4. jitter hurts only mildly at small std;
5. shuffled correspondence destroys the loss immediately.

That pattern means the supervision is carried mostly by **precomputed correspondence + coarse coordinates**,
not by rich local geometry.

## Fast log summary

```bash
python tools/concerto_projection_shortcut/summarize_logs.py exp/concerto/*/train.log
```

It prints a CSV with the last seen `loss`, `enc2d_loss`, `mask_loss`, `unmask_loss`, and `roll_mask_loss`.

## Caveats

- This MVP targets the released `Concerto-v1m1` pretrain path only.
- It probes the **cross-modal enc2d branch**, not the full downstream segmentation or open-vocabulary story.
- A positive result does **not** prove the entire Concerto representation is shortcut-only.
  It proves that the released cross-modal objective admits a much easier path than “full 3D semantics”.
- If this lights up, the next paper-quality step is to run downstream linear probing / language probing
  after retraining under `coord_mlp` or `zero_appearance` conditions.

## Suggested next step after MVP

If `coord_mlp` is strong, the next clean experiment is:

- replace explicit `global_correspondence` with a weaker, noisy, or partially dropped correspondence map;
- or require the model to predict a latent query for 2D alignment instead of indexing teacher patches with
  explicit correspondence.

That would be the Concerto analogue of “do not give the answer token directly”.
