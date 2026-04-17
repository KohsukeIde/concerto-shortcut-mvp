# Official Concerto Large-Video Causal Battery

This battery used `pretrain-concerto-v1m1-2-large-video.pth`, which is the
released full pretraining checkpoint with enc2d heads for the large-video
variant. Treat these numbers as cross-variant evidence, not as the final
Concerto paper main-variant diagnostic.

- raw logs: `/groups/qgah50055/ide/concerto-shortcut-mvp/data/runs/official_causal_battery/official-large-video-step0-mb32/raw.csv`

| dataset | mode | stress | batches | enc2d loss mean | delta vs baseline |
| --- | --- | --- | ---: | ---: | ---: |
| arkit | baseline | clean | 32 | 2.737827 | 0.000000 |
| arkit | global_target_permutation | clean | 32 | 3.324298 | 0.586471 |
| arkit | cross_image_target_swap | clean | 32 | 3.371361 | 0.633534 |
| arkit | cross_scene_target_swap | clean | 32 | 3.324546 | 0.586719 |
| scannet | baseline | clean | 32 | 3.416545 | 0.000000 |
| scannet | global_target_permutation | clean | 32 | 5.466386 | 2.049841 |
| scannet | cross_image_target_swap | clean | 32 | 5.502719 | 2.086174 |
| scannet | cross_scene_target_swap | clean | 32 | 5.467065 | 2.050520 |
