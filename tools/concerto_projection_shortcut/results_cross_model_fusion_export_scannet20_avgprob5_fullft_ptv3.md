# Cross-Model Fusion Saved-Prediction Evaluation

Predictions are saved scene-wise in Pointcept-style `pred/*.npy` and `submit/*.txt` folders, then re-scored from the saved raw-point predictions.

This verifies the save/eval path for fusion predictions. It does not add Pointcept test-time fragment voting; therefore it should match the raw-aligned fusion protocol rather than the `0.8075` Pointcept full-FT precise/test result.

| variant | mIoU | mAcc | allAcc | weak mIoU | picture | p->wall |
|---|---:|---:|---:|---:|---:|---:|
| `fullft_single_saved` | `0.7969` | `0.8779` | `0.9276` | `0.7014` | `0.4296` | `0.4015` |
| `avgprob_all_saved` | `0.8064` | `0.8832` | `0.9321` | `0.7110` | `0.4197` | `0.4680` |
