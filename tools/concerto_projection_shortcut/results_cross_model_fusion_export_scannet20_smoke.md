# Cross-Model Fusion Saved-Prediction Evaluation

Predictions are saved scene-wise in Pointcept-style `pred/*.npy` and `submit/*.txt` folders, then re-scored from the saved raw-point predictions.

This verifies the save/eval path for fusion predictions. It does not add Pointcept test-time fragment voting; therefore it should match the raw-aligned fusion protocol rather than the `0.8075` Pointcept full-FT precise/test result.

| variant | mIoU | mAcc | allAcc | weak mIoU | picture | p->wall |
|---|---:|---:|---:|---:|---:|---:|
| `fullft_single_saved` | `0.4208` | `0.4709` | `0.9585` | `0.4284` | `0.9381` | `0.0124` |
| `avgprob_all_saved` | `0.3906` | `0.4236` | `0.9513` | `0.3356` | `0.0614` | `0.9321` |
