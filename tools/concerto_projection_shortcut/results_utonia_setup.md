# Utonia Setup Note

## Summary

Prepared the official Utonia inference repository locally so it can be used as
an external comparator in the downstream audit.

This started as a setup/integration note and now includes both a successful
one-scene GPU smoke and a completed ScanNet point-stagewise trace.

## Local Repo

- Official repo cloned to:
  - [`external/Utonia`](/groups/qgah50055/ide/concerto-shortcut-mvp/external/Utonia)
- Source:
  - `https://github.com/Pointcept/Utonia`

## Available Local Weights

- Backbone / pretrained checkpoints:
  - [`data/weights/utonia/utonia.pth`](/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/utonia.pth)
  - [`data/weights/utonia/pretrain-utonia-v1m1-0-base_stagev2.pth`](/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/pretrain-utonia-v1m1-0-base_stagev2.pth)
- ScanNet linear head:
  - [`data/weights/utonia/utonia_linear_prob_head_sc.pth`](/groups/qgah50055/ide/concerto-shortcut-mvp/data/weights/utonia/utonia_linear_prob_head_sc.pth)

## Relevant Upstream Entry Points

- Utonia ScanNet semantic segmentation demo:
  - [`external/Utonia/demo/2_sem_seg.py`](/groups/qgah50055/ide/concerto-shortcut-mvp/external/Utonia/demo/2_sem_seg.py)
- Utonia model loader:
  - [`external/Utonia/utonia/model.py`](/groups/qgah50055/ide/concerto-shortcut-mvp/external/Utonia/utonia/model.py)

## Important Upstream Constraint

Utonia's public repo is positioned as an inference/demo repo. The README
explicitly says reproduction of pretraining should be done in the Pointcept
codebase, while this repo is for pretrained weights, quick inference, and
visualization.

For this project, that means the most practical comparator path is:

1. use released Utonia weights;
2. reuse the ScanNet linear-head path already provided upstream;
3. adapt the current downstream-audit protocol around that released artifact.

This is analogous to how the Concerto main-variant audit is anchored to the
released `concerto_base_origin.pth` artifact instead of rerunning main
pretraining from scratch.

## Immediate Integration Plan

1. export point-level predictions/features on the same ScanNet val protocol used
   for Concerto/Sonata/PTv3;
2. run the existing downstream audit on at least:
   - `picture_vs_wall`
   - `door_vs_wall`
   - `counter_vs_cabinet`
3. if stagewise looks stable, add the oracle/actionability battery on the same
   three pairs.

## One-Scene Smoke

ABCI-Q smoke job `134182.qjcm` completed successfully on
[`scene0685_00`](/groups/qgah50055/ide/concerto-shortcut-mvp/data/scannet/val/scene0685_00)
with the released Utonia backbone and released ScanNet linear head.

Smoke facts:

- raw scene points: `132720`
- transformed points after grid sample: `93667`
- feature shape after unpool: `(93667, 1386)`
- logits shape: `(93667, 20)`
- inverse-restored raw prediction shape: `(132720,)`
- single-scene valid-point accuracy: `0.937095`

This confirms that the public Utonia inference stack works against this repo's
raw ScanNet layout and that downstream audit work is blocked only on evaluator
wiring, not on checkpoint or dependency mismatch.

## ScanNet Point-Stagewise Trace

The follow-up ScanNet point-stagewise trace is now complete for:

- `picture_vs_wall`
- `door_vs_wall`
- `counter_vs_cabinet`

Result files:

- [`tools/concerto_projection_shortcut/results_utonia_scannet_point_stagewise_trace/utonia_scannet_point_stagewise_trace.md`](/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_utonia_scannet_point_stagewise_trace/utonia_scannet_point_stagewise_trace.md)
- [`tools/concerto_projection_shortcut/results_utonia_scannet_point_stagewise_trace/utonia_scannet_point_stagewise_trace.csv`](/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_utonia_scannet_point_stagewise_trace/utonia_scannet_point_stagewise_trace.csv)
- [`tools/concerto_projection_shortcut/results_utonia_scannet_point_stagewise_trace/utonia_scannet_point_stagewise_trace_confusion.csv`](/groups/qgah50055/ide/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_utonia_scannet_point_stagewise_trace/utonia_scannet_point_stagewise_trace_confusion.csv)

Headline numbers:

- `picture_vs_wall`: point/logit/direct balanced accuracy
  `0.8847 / 0.9039 / 0.9320`
- `door_vs_wall`: point/logit/direct balanced accuracy
  `0.7294 / 0.9122 / 0.9624`
- `counter_vs_cabinet`: point/logit/direct balanced accuracy
  `0.6740 / 0.8366 / 0.9499`

Interpretation:

- The released Utonia ScanNet stack realizes the audited pairwise information
  much more cleanly than Concerto and Sonata on these pairs.
- In particular, `picture_vs_wall` already has a very strong direct fixed-logit
  margin (`0.9320` balanced accuracy), so the large readout/actionability gap
  observed in Concerto/Sonata is not universal across recent 2D-3D SSL style
  rows.

## Current Status

- Repo cloned: yes
- Local weights present: yes
- One-scene GPU smoke: passed
- Local audit integration: point-stagewise trace completed
- Evaluated results: smoke + ScanNet point-stagewise trace complete
- Oracle/actionability battery: not yet run
