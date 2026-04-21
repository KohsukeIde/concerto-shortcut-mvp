# Cross-Model Downstream Audit Summary

| model | pair | point bal acc | point AUC | logit bal acc | logit AUC | direct bal acc | direct AUC | base IoU | base pos->neg | top2 | top5 | oracle top2 IoU | oracle top5 IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PTv3 supervised ScanNet200 | picture_vs_wall | 0.9097 | 0.9604 | 0.9094 | 0.9537 | 0.8613 | 0.9697 | 0.4624 | 0.2667 | 0.8461 | 0.9168 | 0.7505 | 0.8388 |
| PTv3 supervised ScanNet200 | door_vs_wall | 0.8973 | 0.9288 | 0.8898 | 0.9269 | 0.8931 | 0.9815 | 0.6890 | 0.1050 | 0.9335 | 0.9868 | 0.8671 | 0.9593 |
| PTv3 supervised ScanNet200 | counter_vs_cabinet | 0.9575 | 0.9920 | 0.9649 | 0.9933 | 0.9729 | 0.9956 | 0.4431 | 0.0144 | 0.9474 | 0.9864 | 0.6237 | 0.8112 |
