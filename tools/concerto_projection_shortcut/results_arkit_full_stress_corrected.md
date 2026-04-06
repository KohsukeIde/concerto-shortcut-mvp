# ARKit Full Stress Corrected

`coord_mlp` and `coord_residual_target` below are evaluated with their own matching configs.
The corruption rows are copied from the separate partial stress runs after the causal branch completed.

| checkpoint | clean | local_surface_destroy | z_flip | xy_swap | roll_90_x |
| --- | ---: | ---: | ---: | ---: | ---: |
| arkit-full-causal-baseline | 3.453537 | 3.897785 | 3.980001 | 3.450245 | 3.845464 |
| arkit-full-causal-coord-mlp | 3.536469 | 3.544549 | 3.620394 | 3.537728 | 3.601248 |
| arkit-full-causal-coord-residual-target | 3.628110 | 4.223210 | 4.181930 | 3.624020 | 4.302245 |
| arkit-full-causal-global-target-permutation | 3.404321 | 3.448366 | 3.525088 | 3.401597 | 3.491385 |
| arkit-full-causal-cross-image-target-swap | 2.922118 | 2.921208 | 2.920336 | 2.920222 | 2.919963 |
| arkit-full-causal-cross-scene-target-swap | 3.405929 | 3.397875 | 3.406161 | 3.405618 | 3.406542 |
