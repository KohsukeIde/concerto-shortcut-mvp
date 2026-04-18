# Main-Variant Coord-MLP Rival Fit

| dataset | coord MLP loss | full baseline | target-swap mean | distance to baseline | distance to swap | relative position | gate hint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| arkit | 6.695971 | 6.026655 | 6.853404 | 0.669316 | 0.157433 | 0.809576 | no_go |
| scannet | 6.735274 | 5.615395 | 7.397524 | 1.119879 | 0.662249 | 0.628394 | partial |
| scannetpp | 6.342475 |  |  |  |  |  | no_reference |
| s3dis | 6.395551 |  |  |  |  |  | no_reference |
| hm3d | 6.372810 |  |  |  |  |  | no_reference |
| structured3d | 6.382244 |  |  |  |  |  | no_reference |

Notes:
- `relative_position = (coord_loss - full_baseline) / (target_swap_mean - full_baseline)`.
- Values near 0 mean the coord rival is close to the full head-refit baseline; values near 1 mean it is near target-swap damage.
