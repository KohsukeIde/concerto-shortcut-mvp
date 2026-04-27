# Concerto Shortcut Current Status

This file is the single entry point for the current state of the shortcut
investigation.

## Current Bottom Line

- Strongest supported claim:
  - The released Concerto `enc2d` objective admits a strong coordinate shortcut.
- Scope of that claim:
  - Objective-level evidence on `ARKitScenes`.
  - The ScanNet continuation proxy shows measurable but limited downstream
    relevance.
  - It does not support the stronger claim that most of Concerto downstream gain
    is explained by the coordinate shortcut.
- Current action:
  - The minimal full-removal fix attempt, `projres_v1a`, completed its ABCI-Q
    gate and is no-go on ScanNet linear.
  - The factorized partial-removal follow-up, `projres_v1b`, also completed its
    ABCI-Q smoke, continuation, stress, and ScanNet linear gates.
  - The selective-prior follow-up, `projres_v1c`, completed prior fitting,
    smoke, continuation, stress, and ScanNet linear gates.
  - Result: v1b improves over v1a and `no-enc2d-renorm`, v1c does not improve
    over v1b, and both remain below the original continuation; no strong-go for
    fine-tuning.
  - A point-conscious long-horizon e025 same-stage check completed for original
    and v1b `combo-b075-a001`. v1b essentially ties the same-stage original
    reference but does not beat it: 0.5526 vs 0.5531 ScanNet proxy mIoU.
  - A frozen-feature post-training pilot completed for SPLICE-3D / HLNS on the
    original e025 checkpoint. All three posthoc arms are near the original e025
    ScanNet proxy mIoU without updating the Concerto backbone.
  - The posthoc e025 stress downstream gate also completed. SPLICE-3D and
    Residual Recycling preserve clean mIoU within `-0.005`, but none improves a
    stress condition by `+0.005`; Stage 1 is no-go as a mainline claim.
  - The Step 1 local-geometry smoke for UGNR completed. `geom_local9` concat
    and residual-expert variants do not improve the offline frozen-cache
    baseline by the required `+0.003` mIoU, so UGNR should not be launched with
    this descriptor.
  - The attempted fresh same-stage e050 original/v1b runs did not produce valid
    e050 checkpoints. Both hit walltime at epoch 6 after delayed node startup.
  - Step 0 official-checkpoint causal battery completed on the released
    `pretrain-concerto-v1m1-2-large-video.pth` checkpoint for ARKit val and
    Concerto ScanNet val. Target swaps strongly increase enc2d loss on both
    datasets, so the signal is not an ARKit-continued artifact. This checkpoint
    is the large-video variant, so it is now treated as cross-variant evidence,
    not as the final Concerto paper main-variant diagnostic.
  - Main-variant Step 0/0.5 completed with `concerto_base_origin.pth` as a
    frozen backbone plus short-refit enc2d alignment projection on the six
    indoor Concerto datasets. Target swaps remain clearly positive on ARKit and
    ScanNet. The coord-MLP rival is weak on ARKit and partial on ScanNet, so it
    is usable only as a lower-bound rival, not as a full shortcut proxy.
    A six-dataset recalibration of the existing coord-MLP losses against the
    completed causal battery is now complete. The original table had positive
    clean-to-corruption closure on `5/6` datasets, with `s3dis` worse than the
    mean corruption reference. A dedicated S3DIS high-validation follow-up
    showed that this negative closure was mainly a tiny validation-cache
    artifact: replacing S3DIS with the high-val run gives positive closure on
    `6/6` datasets and mean closure `28.9%`. S3DIS remains weak (`13.3%`), so
    the paper-safe claim is still dataset-dependent coordinate-satisfiable
    signal, not a uniformly strong six-dataset coord-only explanation.
  - A target-corruption distance diagnostic completed for the main-variant
    ARKit/ScanNet battery. ARKit cross-scene targets are not closer to the
    original targets than cross-image targets, so the lower ARKit cross-scene
    loss is not explained by `cos(t_original, t_corrupted)` alone.
  - SR-LoRA v5 Phase A completed its code smoke, 256-iteration pilot, 4-arm
    training matrix, and representative downstream follow-ups. Training is
    stable, but downstream is no-go. `m=0.1` gives only `+0.0009/+0.0015`
    ScanNet linear last/best mIoU and no stress gain; `m=0.2` increases margin
    pressure but gives `-0.0003/-0.0003` linear last/best mIoU and no stress
    gain. Do not launch the remaining matrix follow-ups without a new
    hypothesis. A loss-based hinge switch has been added and smoke-tested in
    `133137.qjcm`; it is wired correctly but not yet downstream-validated.
  - Concerto 3D patch-separation Step A, exact-patch controls, and point-level
    stage-wise traces completed. The first-pass DINO Step A' vs Concerto A''
    comparison was not apples-to-apples: on the same image-patch subset,
    `picture_vs_wall` is DINO `0.6146`, encoder-pooled `0.5548`, patch-proj
    `0.5359`, and pooled linear logits `0.7128` balanced accuracy. The
    point-level ScanNet trace shows the frozen backbone already separates the
    weak pairs much better (`picture_vs_wall` point feature `0.7041`, linear
    logits `0.7602`; most other pairs `>=0.86`). The bottleneck is therefore
    not generic DINO-to-3D semantic loss; it is pair/stage/subset-specific, with
    `picture -> wall` still present at the final 20-way readout. This
    supersedes the earlier raw comparison of DINO `0.7797` against Concerto
    `0.5381/0.5547` as an immediate semantic-transfer-bottleneck claim.
  - ScanNet decoder-probe per-class Stage 1 completed on
    `concerto_base_origin.pth` with the frozen encoder and trainable decoder
    config family. The 100 epoch run finished successfully (`133217.qjcm`,
    `rt_QF=2`, `00:34:04`) with final precise eval
    `mIoU/mAcc/allAcc = 0.7888/0.8813/0.9243`, but `picture` remains low at
    `0.4217` IoU and `43.1%` of target `picture` points are still predicted
    as `wall`. This weakens the cheapest explanation that the `picture -> wall`
    bottleneck is solved by simply replacing the linear probe with the Concerto
    decoder probe. A follow-up origin-only stage-wise trace on the decoder
    checkpoint shows `picture_vs_wall` is separable in point features
    (`0.8376` balanced accuracy), but the direct 20-way class margin is much
    weaker (`0.7203`) and sampled target `picture` points still go to `wall`
    `54.96%` of the time. The official-like origin full-FT run also completed
    at `data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800` with final
    mIoU/mAcc/allAcc `0.8075 / 0.8838 / 0.9309`; `picture` improves to
    `0.4415`, while `door/cabinet/counter/sink` land at
    `0.7695 / 0.7856 / 0.7270 / 0.7473`. The pairwise stagewise/oracle audit on
    this full-FT checkpoint is now also complete: `picture_vs_wall` reaches
    point/logit/direct balanced accuracy `0.7175 / 0.7206 / 0.7052`, while the
    base oracle-analysis row is `mIoU=0.7972`, `picture=0.4338`, and
    `picture -> wall=0.3956`; oracle top-2 / top-5 raise `picture` to
    `0.8304 / 0.9567`. This removes the stale "full FT still pending" gap and
    shows that full fine-tuning reduces but does not erase the
    readout/actionability gap. Details are in
    `tools/concerto_projection_shortcut/results_scannet_origin_fullft.md`,
    `tools/concerto_projection_shortcut/results_scannet_origin_fullft_point_stagewise_trace/scannet_point_stagewise_trace.md`,
    and
    `tools/concerto_projection_shortcut/results_scannet_origin_fullft_oracle_actionability/oracle_actionability_analysis.md`.
  - Same-checkpoint confusion-graph residual readout pilot completed on the
    origin decoder checkpoint. The implementation works, but the naive
    antisymmetric logit correction is no-go: multi-pair best mIoU delta is only
    `+0.0002`, and a refined `picture:wall` only sweep improves `picture` IoU
    by only `+0.00055`. This supports readout/calibration as a bottleneck, but
    the simple post-hoc residual expert is not a sufficient positive method.
  - Same-checkpoint top-K pairwise reranking decoder completed on the origin
    decoder checkpoint. The reranker fits the train candidate problem
    (`candidate base acc 0.9600` to `rerank acc 0.9868` at step 2000). Large
    lambdas overcorrect, but a small-lambda follow-up finds a tiny positive:
    deterministic offline base `mIoU=0.7789`, `picture IoU=0.4096`;
    `topk2_lam0p05` gives `mIoU=0.7791` (`+0.00022`) and `picture IoU=0.4102`
    (`+0.00066`). This is below gate and not paper-relevant yet, but it shows
    the readout-side lever is real and very small.
  - Oracle/actionability analysis completed on the same origin decoder
    checkpoint. It shows large candidate-set headroom: `picture` top-2 hit rate
    is `0.8929` and top-5 hit rate is `0.9599`; oracle top-2 raises mIoU from
    `0.7778` to `0.9197` and `picture` IoU from `0.4034` to `0.8579`.
    However, train-derived proxies fail: `pair_probe_top2` drops mIoU to
    `0.7567` and `picture` IoU to `0.1722`, while class-prior/bias calibration
    also hurts. Conclusion: readout headroom is real, but unconstrained
    post-hoc readout methods are miscalibrated; the next positive method needs
    held-out-train validation or a lightweight decoder adaptation, not another
    free reranker.
  - Validation-aware constrained Top-K set decoder completed on the same origin
    decoder checkpoint. The first capped run is invalid for selection because
    heldout rare classes were missing. The full-scan run is valid and selects
    `k2_lam0p2_tau0p5_gap999` on heldout train. On ScanNet val it gives only a
    weak positive: mIoU `0.77865983 -> 0.77878256` (`+0.00012274`),
    `picture` IoU `0.40257231 -> 0.40386984` (`+0.00129753`), and
    `picture -> wall` `0.43864309 -> 0.42774366`. This is directionally
    consistent with the oracle/actionability diagnosis but far below gate. Do
    not spend more runs on offline fixed-logit reranking without a stronger
    new constraint or adaptation hypothesis.
  - CoDA decoder-adapter pilot completed on the same origin decoder checkpoint.
    This changes the family from fixed-logit reranking to a trainable residual
    feature-to-logit map `z = z0 + A(h)` with weighted CE, confusion-pair CE,
    KL-to-base, and residual L2. It trains successfully and improves heldout
    train, but the heldout-selected aggressive variant overcorrects ScanNet val
    (`mIoU -0.0114`, `picture -0.0283`). An all-variant val sweep finds only a
    tiny safe positive: best mIoU `+0.00023654`, best safe `picture` delta
    `+0.00167642`. Current CoDA is no-go as a paper-relevant positive. The
    oracle headroom is still real, but cached-feature post-hoc adapters are not
    recovering it.
  - CoDA transfer-failure analysis completed. It shows why heldout selection
    fails: heldout `picture` is easy and train-like (`GT top1=0.9259`,
    `picture-wall margin=+4.08`), while val `picture` is shifted and
    wall-dominated (`GT top1=0.5357`, `picture-wall margin=-0.342`). The
    aggressive adapter moves val target `picture` in the intended direction
    (`picture->wall 0.4388 -> 0.3486`) but overpredicts `picture` and hurts
    other classes, so `picture` IoU drops (`0.4022 -> 0.3776`) and mIoU drops.
    Conclusion: the class-balanced cached heldout objective is not a
    representative selection criterion. Do not broaden diagnosis further; next
    method must change the protocol, e.g. in-loop decoder adaptation with real
    augmentation or stricter full-distribution calibration.
  - CIDA in-loop decoder adaptation completed on the same origin decoder
    checkpoint. This moves from cached-feature post-hoc correction to training
    the decoder/head inside the real ScanNet augmentation loop with frozen
    encoder, weak-class CE, confusion-pair CE, KL-to-base, and prediction
    distribution anchoring. The train path is stable after matching Pointcept's
    standard decoder-probe train-mode behavior. However, batch-size-1 full val
    is no-go: CIDA-base gives mIoU `0.7752` (`-0.0031`) and `picture` IoU
    `0.3856` (`-0.0203`); CIDA-strong gives mIoU `0.7746` (`-0.0037`) and
    `picture` IoU `0.3863` (`-0.0196`). Both reduce `picture -> wall`
    slightly (`~0.439 -> ~0.430`) but damage the broader weak-class decision
    surface. Do not continue this exact pair-emphasis CIDA line without a new
    full-multiclass preservation constraint.
  - Plain same-checkpoint origin LoRA per-class control completed, and the
    matched no-LoRA same-head baseline has now also completed. Both use
    `concerto_base_origin.pth` through `DefaultLORASegmentorV2` with the same
    PTv3 base encoder-mode linear-head family. The no-LoRA linear-head baseline
    gives mIoU `0.7617`; the LoRA run gives `0.7749`, so LoRA is positive in
    the matched head family (`+0.0132` mIoU). It also moves the target failure
    in the intended direction relative to same-head no-LoRA: `picture` IoU
    improves `0.4078 -> 0.4303`, and `picture -> wall` drops
    `0.4151 -> 0.3867`. The earlier comparison against decoder probe
    (`0.7888`) was head-capacity confounded; LoRA remains below that stronger
    decoder-probe reference, but it should no longer be described as simply
    damaging aggregate performance. Details are in
    `tools/concerto_projection_shortcut/results_scannet_lora_origin_perclass.md`
    and
    `tools/concerto_projection_shortcut/results_scannet_lora_origin_same_head_perclass.csv`.
  - Decoder-capacity-matched origin LoRA control completed. This uses the
    same origin decoder-probe family as `scannet-dec-origin-e100` and adds
    rank-8 qkv LoRA, with the plain decoder-probe semseg objective and no
    weak-class or pairwise losses. The result is near-tie but not positive
    against the decoder baseline: mIoU `0.7888 -> 0.7860`,
    `picture` IoU `0.4217 -> 0.4204`, and `picture -> wall`
    `0.4310 -> 0.4387`. Therefore, the matched-head linear LoRA gain does not
    clearly survive decoder-capacity matching in this plain setup. Details are
    in
    `tools/concerto_projection_shortcut/results_scannet_dec_lora_origin_perclass.md`
    and
    `tools/concerto_projection_shortcut/results_scannet_dec_lora_origin_perclass.csv`.
  - Official Concerto LoRA recipe check completed. The upstream Pointcept
    recipe `semseg-ptv3-large-v1m1-0f-scannet-ft-lora.py` is materially
    different from the local decoder+LoRA control: it is a PTv3 large
    encoder-mode linear-head LoRA recipe (`backbone_out_channels=1728`,
    `enc_mode=True`, no decoder branch, 800 epochs, batch size 24, AdamW
    lr `0.002`). The local decoder+LoRA no-go therefore should not be reported
    as an official Concerto LoRA no-go. The required upstream checkpoint
    `exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth`
    is not currently present locally. Details are in
    `tools/concerto_projection_shortcut/results_official_lora_recipe_check.md`.
  - Retrieval/prototype readout follow-up completed on the same origin decoder
    checkpoint. Prototype and multi-prototype variants slightly reduce
    `picture -> wall` but do not recover oracle headroom: best mIoU is only
    `+0.0002` and best safe `picture` gain is only `+0.0008`. A small kNN
    retrieval pass is also no-go: best mIoU is `+0.0002`, best `picture` gain
    is `+0.0003`, and `picture -> wall` drops by only `0.0052`. Treat
    retrieval/prototype as no-go under this protocol. Details are in
    `tools/concerto_projection_shortcut/results_prototype_readout.md` and
    `tools/concerto_projection_shortcut/results_knn_readout_small.md`.
  - LP-FT / class-safe LoRA follow-up completed in the origin linear-head
    family. LP-FT plain warm-starts from the same-head linear probe and is a
    tiny mIoU positive over the prior plain LoRA control (`0.7749 -> 0.7771`),
    but it does not improve the central `picture -> wall` failure
    (`0.3867 -> 0.4222`) and `picture` IoU is slightly lower
    (`0.4303 -> 0.4275`). Class-safe LoRA reduces `picture -> wall` to
    `0.3554`, but hurts mIoU (`0.7706`) and `picture` IoU (`0.4077`), so it is
    no-go with the current weights. Current decision: retrieval/prototype
    no-go, class-safe no-go, LP-FT warm start only a weak positive control; do
    not launch more broad method sweeps without a sharper hypothesis. Details
    are in
    `tools/concerto_projection_shortcut/results_scannet_lora_lpft_classsafe.md`.
  - Decoupled classifier readout pilot completed on the same origin decoder
    checkpoint. This tests the long-tail / classifier-retraining family that
    had not been covered by prior post-hoc, retrieval, LoRA, CoDA, or CIDA
    runs. The corrected run uses a class-balanced feature bank for cRT but
    raw train priors from 256 ScanNet train batches for logit adjustment and
    Balanced Softmax. Result: no-go. `tau` normalization moves mIoU by only
    `+0.0002` and worsens `picture`; direct cRT / Balanced Softmax severely
    overcorrect (`mIoU ~0.666`, `picture ~0.104`); small trust-region mixing
    reduces `picture -> wall` by about `0.026` but does not improve `picture`
    IoU or weak-class mean. Treat offline decoupled classifier learning as
    another negative under this protocol. Details are in
    `tools/concerto_projection_shortcut/results_decoupled_classifier_readout.md`.
  - Latent-subgroup decoder pilot completed on the same origin decoder
    checkpoint. The diagnostic supports a real `picture` subgroup shift:
    the dominant val `picture` cluster has `target_top1=0.2737`,
    `wall_top1=0.7172`, and mean `picture-wall` margin `-3.2460`, while
    train/heldout include much easier `picture` clusters. However, targeted
    sub-center interpolation still does not recover the oracle/actionability
    headroom. Best variant `subcenter_tau0p1_lam0p4` gives only
    mIoU `0.7777 -> 0.7788` (`+0.0011`) and `picture` IoU
    `0.4043 -> 0.4060` (`+0.0017`), though it reduces `picture -> wall`
    `0.4348 -> 0.4096`. This is below gate, so treat latent-subgroup readout
    as diagnostic-positive but method-no-go under the current offline
    protocol. Details are in
    `tools/concerto_projection_shortcut/results_latent_subgroup_decoder.md`.
  - Region / superpoint diagnostic completed on the same origin decoder
    checkpoint. This tests the newer granularity hypothesis without training:
    coarse voxel regions are formed on ScanNet val and point logits are
    compared to region-averaged logits, region-majority oracle labels, and
    region top-k oracles. The result is diagnostic-positive but not a direct
    method: small regions preserve large oracle headroom (`region_oracle_s4_top5`
    mIoU `0.9783`, `picture` `0.9462`, close to point top-5 oracle), but
    actual region-logit smoothing hurts (`region_logits_s4` mIoU `0.7695`,
    `picture` `0.3974`; coarser regions are much worse). For `picture`,
    point top-2/top-5 hits are `0.8954/0.9615`; region top-2/top-5 are
    similar at fine granularity (`s4`: `0.9009/0.9624`) but degrade as regions
    get coarser. Region purity also drops with granularity (`picture`
    mean purity `0.9182` at `s4`, `0.5614` at `s32`), and wall-majority
    picture regions rise (`0.0617 -> 0.3556`). Interpretation: object/region
    granularity matters, but naive region averaging is not the method; a
    future region-level method would need object-mask-quality regions or
    learned region proposals rather than coarse smoothing. Details are in
    `tools/concerto_projection_shortcut/results_region_superpoint_analysis.md`.
  - Purity-aware hybrid region decoder (PHRD) zero-train gate completed on the
    same origin decoder checkpoint. Label-free proxies are useful for general
    region purity (`AUC purity>=0.9` around `0.87-0.89`), but they are not a
    safe positive gate for `picture`: at fine scale (`s4`), confidence/entropy
    proxies strongly identify hard `picture->wall` regions (`AUC hard
    picture-wall` up to `0.93`) and have negative correlation with
    picture-specific purity. The high-proxy mixing sweep is no-go (best
    mIoU only `+0.0002`, picture does not improve). The inverse low-proxy
    follow-up is also no-go (best mIoU `+0.0008`, best safe picture
    `+0.0019`). Conclusion: purity-aware coarse-region logit mixing is not a
    positive method. The useful finding is mechanistic: generic confidence
    proxies are anti-aligned with the hard wall-adjacent `picture` subgroup, so
    a serious region method needs learned/object-quality proposals or
    class-specific masks rather than label-free region averaging. Details are
    in `tools/concerto_projection_shortcut/results_purity_aware_region_readout.md`
    and
    `tools/concerto_projection_shortcut/results_purity_aware_region_readout_lowgate.md`.
  - Proposal recall analysis completed as a minimal gate for proposal-first /
    mask-lite methods. Fine voxel proposals preserve high-purity candidate
    recall: for `picture`, `voxel s4` at purity `>=0.9` covers `0.8055` of
    `picture` points while touching only `0.0142` of all val points; `voxel s8`
    still covers `0.6677`. However, connected same-base-prediction components
    collapse for `picture` (`pred_cc s4` recall `0.3188`, `s8` `0.2077`) and
    show high wall contamination (`0.3965/0.4265`). Interpretation: proposal
    candidates exist at fine granularity, but base-pred connected components
    already merge hard `picture` into `wall`; any PVD-style method needs a
    learned object-quality verifier/mask, not base-pred region merging. Details
    are in
    `tools/concerto_projection_shortcut/results_proposal_recall_analysis.md`.
  - Proposal-then-Verify Decoder (PVD) pilot completed on the same origin
    decoder checkpoint. A lightweight MLP verifier trained on `s4` proposals
    gets reasonable proposal-level accuracy on val (`picture` proposal acc
    `0.6587`, most other hard/counterpart classes `0.87-0.94`), but logit
    fusion is no-go. Best mIoU remains the base decoder (`0.7789`); the safest
    PVD variant (`thr=0.9,beta=0.25`) reduces `picture -> wall`
    `0.4401 -> 0.4024` but lowers `picture` IoU `0.4061 -> 0.3952` and weak
    mean. Interpretation: a verifier can find some hard-class proposals, but
    class-level proposal boosts still damage the multiclass decision geometry.
    The next region-family method would need point-mask assignment or a
    proposal-native classifier, not scalar logit boosting. Details are in
    `tools/concerto_projection_shortcut/results_proposal_verify_decoder.md`.
  - Masking ranking battery completed for the same origin decoder and linear
    checkpoints, plus train-majority, coord-only, and class-balanced coord-only
    baselines in the same voxel-level evaluation space. Concerto decoder and
    linear readouts both keep high mIoU under heavy sparsity: decoder
    `0.7874 -> 0.7636` and linear `0.7691 -> 0.7589` under random keep `0.2`;
    structured block keep `0.2` is also high at decoder `0.7569` and linear
    `0.7551`. Feature-zero still collapses both (`0.0680` decoder, `0.0390`
    linear), so this is not a pure coordinate-only story. The explicit
    coord-only baselines also rule that out as a sufficient explanation:
    ordinary coord MLP clean mIoU is only `0.0726`, class-balanced coord MLP is
    `0.0707`, and train-majority wall is `0.0151`. Clean-to-masked ranking
    shift is now measurable with the supervised comparator but still not a full
    protocol critique. The valid PTv3 v1.5.1 compatibility row gives clean
    `0.7697`, random keep `0.2` `0.7143`, structured keep `0.2` `0.6521`, and
    feature-zero `0.0269`. Concerto decoder/linear retain more mIoU under
    random/structured sparsity than this supervised PTv3 row, while feature-zero
    still collapses all model rows. Current interpretation: strong
    sparsity-tolerance evaluation signal and anti coord-only-baseline evidence,
    plus a supervised-comparator ranking signal, but still not shortcut-proof
    task-level evidence by itself. Details are in
    `tools/concerto_projection_shortcut/results_masking_ranking_battery.md`.
  - Downloaded comparator check completed for released PTv3 supervised,
    PTv3-PPT, Sonata, Concerto-head, and Utonia-head weights. The PTv3
    checkpoints download and load with `missing=0/unexpected=0`, but their clean
    mIoU under the current repo/data protocol is far below expected supervised
    levels (`0.1496` for PTv3 base, `0.0422` for PTv3-PPT). The downloaded
    original PTv3 config also fails in the current code due the removed
    `cls_mode` argument, supporting a version/protocol mismatch interpretation.
    The released Sonata backbone and ScanNet linear head can be merged into a
    usable external SSL comparator: clean `0.7169`, random keep `0.2` `0.6942`,
    structured keep `0.2` `0.6752`, feature-zero `0.0607`. This adds an
    external SSL row but still shows weak clean-to-masked ranking shift. Details
    are in
    `tools/concerto_projection_shortcut/results_masking_downloaded_comparators.md`.
  - Utonia external comparator setup advanced from static prep to executable
    integration and then to a completed ScanNet point-stagewise trace. The
    official inference repo was cloned to `external/Utonia`, local released
    weights are available under `data/weights/utonia/`, and an ABCI-Q
    one-scene ScanNet smoke (`134182.qjcm`) succeeded with the released
    backbone plus released ScanNet linear head on `scene0685_00`: raw
    `132720` points, transformed `93667` points, unpooled feature shape
    `(93667, 1386)`, logits shape `(93667, 20)`, inverse-restored raw
    prediction shape `(132720,)`, and single-scene valid accuracy `0.937095`.
    The follow-up point-stagewise trace is now complete for
    `picture_vs_wall`, `door_vs_wall`, and `counter_vs_cabinet`. Utonia's
    released ScanNet stack shows a much stronger fixed-readout realization on
    these audited pairs than Concerto or Sonata: `picture_vs_wall` reaches
    point/logit/direct balanced accuracy `0.8847 / 0.9039 / 0.9320`,
    `door_vs_wall` reaches `0.7294 / 0.9122 / 0.9624`, and
    `counter_vs_cabinet` reaches `0.6740 / 0.8366 / 0.9499`. This makes
    Utonia a constructive external comparator showing that the large
    readout/actionability gap seen in Concerto and Sonata is not universal
    across recent 2D-3D SSL style rows. Details are in
    `tools/concerto_projection_shortcut/results_utonia_setup.md` and
    `tools/concerto_projection_shortcut/results_utonia_scannet_point_stagewise_trace/utonia_scannet_point_stagewise_trace.md`.
  - The follow-up Utonia oracle/actionability battery is now also complete.
    Base Utonia is already much cleaner than Concerto/Sonata on the audited
    weak pairs (`picture` base IoU `0.2952`, `picture -> wall 0.1284`,
    `picture` top-1/top-2/top-5 hit `0.8716 / 0.9994 / 1.0000`), but it still
    leaves residual headroom: oracle top-2 / top-5 raise `picture` to
    `0.9747 / 1.0000` and overall mIoU to `0.9367 / 0.9908`. This sharpens the
    cross-model reading: large readout/actionability gaps are not universal
    across recent 2D-3D SSL-style rows, but residual actionability gap is still
    present even in the constructive Utonia comparator. Details are in
    `tools/concerto_projection_shortcut/results_utonia_scannet_oracle_actionability/oracle_actionability_analysis.md`
    and the updated
    `tools/concerto_projection_shortcut/results_cross_model_downstream_audit_scannet20.md`.
  - Utonia ScanNet20 support-stress battery is now complete. With full-scene
    nearest-neighbor scoring from retained support logits, Utonia gives clean
    `mIoU=0.7576`, random keep `0.2` `0.7469` (`-0.0107`), structured block
    keep `0.2` `0.2834` (`-0.4742`), object-style masked-model keep `0.2`
    `0.2360` (`-0.5216`), fixed `4000` points `0.4190` (`-0.3386`), and
    feature-zero `0.7472` (`-0.0104`). Reading: Utonia's cleaner fixed
    readout does not eliminate support redundancy; random keep `0.2` remains a
    weak stress, while structured/object-style missing-support is severe. The
    earlier note that the public Utonia path is "raw-feature agnostic" was too
    strong: Utonia's default transform explicitly constructs
    `feat=(coord,color,normal)`, while the model also receives `coord` and
    `grid_coord` as structural keys. The low `feature_zero` damage should be
    treated as an audited low-sensitivity finding under this released stack,
    not as evidence that raw features are omitted from the input path.
    Details are in
    `tools/concerto_projection_shortcut/results_utonia_scannet_support_stress/utonia_scannet_support_stress.md`
    and the full channel audit in
    `tools/concerto_projection_shortcut/results_utonia_scannet_support_stress_featurezero_audit/utonia_scannet_support_stress.md`.
  - The follow-up Utonia feature-channel audit (`135554.qjcm`) is complete.
    It separates the legacy all-`feat` zero row from `feat_zero_color_normal`,
    `feat_zero_coord`, and raw `--wo_color/--wo_normal`-style ablations.
    Full 312-scene values: `feat_zero_color_normal=0.7475` (`-0.0101`),
    `feat_zero_coord=0.7586` (`+0.0010`), `raw_wo_color=0.7557`
    (`-0.0019`), `raw_wo_normal=0.7526` (`-0.0050`), and
    `raw_wo_color_normal=0.7477` (`-0.0099`). The correct reading is not that
    Utonia omits raw features; the released stack explicitly accepts
    `feat=(coord,color,normal)`, but its ScanNet prediction is only weakly
    sensitive to these channel-zero / raw-missing ablations.
  - Paper-facing award-level consolidation tables are now generated. The
    completed six-dataset main-variant causal battery is reformatted as
    `tools/concerto_projection_shortcut/results_main_variant_causal_battery_paper_table.md`,
    the six-dataset coord-MLP rival calibration is summarized in
    `tools/concerto_projection_shortcut/results_coord_mlp_rival_six_dataset_calibration.md`,
    and the scene/object binding-profile rows are consolidated in
    `tools/concerto_projection_shortcut/results_binding_profile_summary.md`.
    The object-side pretext logs for PointGPT-S no-mask and no-mask
    order-random are similarly summarized in
    `3D-NEPA/results/pointgpt_object_pretext_summary.md`. This closes the
    "results exist but are not paper-readable" gap for the main causal table,
    unified binding profile, and object pretext summary.
  - Scene support-stress severity curves are now complete for Concerto decoder,
    Concerto linear, Sonata linear, PTv3 ScanNet20, PTv3 ScanNet200, and PTv3
    S3DIS. The central pattern is stable: random keep20 is materially weaker
    than structured/object-style missing support, especially under full-scene
    scoring. Full-scene random keep20 damage / structured keep20 damage:
    Concerto-D `0.0342 / 0.4920`, Concerto-L `0.0180 / 0.4688`, Sonata-L
    `0.0297 / 0.4403`, PTv3 ScanNet20 `0.0750 / 0.5212`, PTv3 ScanNet200
    `0.0829 / 0.2562`, PTv3 S3DIS `0.2661 / 0.4486`. Object-style keep20 is
    also severe: Concerto-D `0.5973`, Concerto-L `0.5735`, Sonata-L `0.5376`,
    PTv3 ScanNet20 `0.6325`, PTv3 ScanNet200 `0.2995`, PTv3 S3DIS `0.5902`.
    The outputs are
    `results_support_severity_concerto_decoder.md`,
    `results_support_severity_concerto_linear.md`,
    `results_support_severity_sonata_linear.md`,
    `results_support_severity_ptv3_scannet20.md`,
    `results_support_severity_ptv3_scannet200.md`, and
    `results_support_severity_ptv3_s3dis.md`.
  - Utonia full severity-curve rerun (`135562.qjcm`) is complete. Clean mIoU is
    `0.7580`; random keep80/50/20/10 gives
    `0.7579 / 0.7581 / 0.7469 / 0.7230`; structured keep80/50/20/10 gives
    `0.6854 / 0.5272 / 0.2900 / 0.1912`; object-style masked-model keep50/20/10
    gives `0.3935 / 0.2241 / 0.1480`; fixed 16k/8k/4k gives
    `0.7122 / 0.6230 / 0.4243`; feature-zero gives `0.7477`.
    This confirms the same qualitative pattern as the rest of the scene-side
    battery: random point sparsity is a weak stress, while structured and
    object-style missing support are severe. The low feature/channel-zero
    sensitivity remains a released-stack Utonia finding rather than evidence
    that raw features are omitted.
  - Representation-readout actionability gap tables are now generated. The
    cross-model, multi-pair table
    `tools/concerto_projection_shortcut/results_actionability_gap_cross_model_pairs.md`
    separates feature content, fixed pair-margin realization, observed
    target-to-confusion error, and oracle top-k headroom. The key reading is
    that the failure is not representation collapse, but also not a trivial
    readout-only artifact: pairwise information is present, Concerto/Sonata
    show large `picture -> wall` confusion, PTv3/Utonia are cleaner on fixed
    pairwise readout, and oracle headroom remains in all rows. The companion
    structural-test table
    `tools/concerto_projection_shortcut/results_readout_fix_structural_test_battery.md`
    summarizes fixed-logit, cached-feature, in-loop, nonparametric,
    decoupled-classifier, region/proposal, LoRA, and full-FT attempts. These
    recover little of the oracle headroom, so the safe phrasing is
    "representation-readout actionability gap", not "readout problem only".
  - Utonia ablation checkpoint availability was checked locally and against the
    public HuggingFace model repo. Only the released stack weights are
    available (`utonia.pth`, `pretrain-utonia-v1m1-0-base_stagev2.pth`, and
    `utonia_linear_prob_head_sc.pth`); no scale/data/design ablation weights
    were found. Therefore Utonia should be used as a constructive released-stack
    comparator, not as causal evidence for RoPE, modality blinding, perceptual
    granularity rescale, scale, or data size individually. Details are in
    `tools/concerto_projection_shortcut/results_utonia_ablation_availability.md`.
  - Cross-model complementarity / simple-fusion audit completed for Concerto
    decoder, Sonata linear, and the released Utonia ScanNet stack on raw-point
    aligned ScanNet20 val. The oracle upper bound is high: Concerto+Utonia
    reaches `0.8301` mIoU, Concerto+Sonata reaches `0.8331`, and the three-model
    oracle reaches `0.8556`. This means the model errors are materially
    complementary. However, simple probability averaging / temperature averaging
    / max-confidence gates recover only a small part of this headroom; the best
    non-oracle row is Concerto+Utonia temperature-averaged probabilities at
    `0.7867` mIoU, above Concerto decoder (`0.7782`) but below the full-FT
    reference (`~0.8075`). Reading: cross-model complementarity is real, but
    SOTA-oriented fusion requires a learned gate/decoder rather than naive
    averaging. Details are in
    `tools/concerto_projection_shortcut/results_cross_model_fusion_scannet20.md`.
  - A bounded learned-fusion pilot was also completed with a two-fold
    scene-level validation stacker over Concerto/Sonata/Utonia probability
    features. This is intentionally not a final train-split method baseline;
    it tests whether logit/probability-level learning can extract the oracle
    complementarity at all. The best CV stacker uses inverse-sqrt class weights
    and reaches `0.7910` mIoU (`picture=0.3836`, `picture->wall=0.3714`),
    compared with `0.7845` for three-model averaging and `0.7796` for Concerto
    decoder in the same CV pass. Reading: learned logit-level fusion extracts
    a small additional gain but remains far below the `0.8556` three-model
    oracle and below the full-FT reference, so clean SOTA likely requires
    feature-level fusion or a stronger train-split multi-expert decoder rather
    than a simple stacker. Details are in
    `tools/concerto_projection_shortcut/results_cross_model_fusion_cv_stacker_scannet20.md`.
  - PTv3 v1.5.1 raw-point probability caches were exported separately from the
    official Pointcept v1.5.1 code path, then added as a fourth expert to the
    cross-model fusion audit. The 4-expert oracle
    (Concerto/Sonata/Utonia/PTv3) reaches `0.8843` mIoU, with weak-class mean
    `0.8221`, so the complementarity ceiling is very large. Pairwise oracles
    also rise: Concerto+PTv3 reaches `0.8485`, Utonia+PTv3 `0.8495`. However,
    simple probability fusion still does not convert this into a SOTA row:
    the best simple 4-expert average is `0.7959`, still below the full-FT
    reference (`~0.8075`). Details are in
    `tools/concerto_projection_shortcut/results_cross_model_fusion_scannet20_with_ptv3.md`.
  - An oracle-guided expert-router pilot was run on the same 4 experts using a
    two-fold scene-level validation protocol. The router target chooses the
    correct expert with the highest correct-class probability, or the expert
    with the highest correct-class probability when all experts are wrong.
    This directly tests whether the large oracle headroom is selectable from
    expert probability features. It is not: the best CV linear stacker is
    `0.7975`, three/four-expert averaging is `0.7959`, the soft oracle-router
    is only `0.7812`, and the hard router is `0.7676`. Reading: complementarity
    is real, but logit/probability-level expert selection is not sufficiently
    learnable. A SOTA fusion attempt would need feature-level fusion or a
    train-split multi-expert decoder, not another probability-level router.
    Details are in
    `tools/concerto_projection_shortcut/results_cross_model_fusion_cv_router_scannet20_with_ptv3.md`.
  - Concerto full-FT was added as an additional cached expert using
    `data/runs/scannet_semseg_origin/exp/scannet-ft-origin-e800/model/model_best.pth`.
    The full-FT raw-probability export loaded cleanly (`missing=0`,
    `unexpected=0`) and produced 312 raw-point aligned ScanNet20 validation
    caches. With Concerto decoder, Sonata, Utonia, Concerto full-FT, and PTv3,
    the 5-expert oracle reaches `0.8969` mIoU (`weak_mean=0.8399`,
    `picture=0.6397`, `picture->wall=0.2873`). Pairwise oracle rows are also
    high: Utonia+fullFT `0.8547`, decoder+fullFT `0.8492`,
    fullFT+PTv3 `0.8429`. The best simple non-oracle row is the 5-expert
    average at `0.8065`, just below the local full-FT reference (`~0.8075`) and
    above the full-FT single row in the same raw-aligned fusion protocol
    (`0.7969`). Thus full-FT raises the practical fusion floor and confirms
    substantial cross-protocol complementarity, but simple logit/probability
    fusion still does not robustly clear the SOTA target. Details are in
    `tools/concerto_projection_shortcut/results_cross_model_fusion_scannet20_with_fullft_ptv3.md`.
  - A matching 5-expert two-fold CV stacker/router pilot was run. The best
    row is still simple averaging: `avgprob_all=0.8064`. CV linear stackers are
    slightly lower (`0.8056` with inverse-sqrt class weights, `0.8051`
    unweighted), the soft oracle-router is `0.8015`, and the hard router is
    `0.7924`. Router targets are dominated by the full-FT expert but still
    assign non-trivial mass to PTv3, Utonia, Concerto decoder, and Sonata,
    indicating real complementarity but poor selectability from probability
    features. Reading: the full-FT expert makes cross-model fusion promising,
    but the next SOTA attempt should move to feature-level fusion or a proper
    train-split multi-expert decoder; probability-level routing is saturated.
    Details are in
    `tools/concerto_projection_shortcut/results_cross_model_fusion_cv_router_scannet20_with_fullft_ptv3.md`.
  - Selective-deferral recoverability/predictability was then tested with
    Concerto full-FT as the default expert and Concerto decoder, Sonata, Utonia,
    and PTv3 as auxiliary experts. Recoverability exists but is class- and
    expert-specific: for example, Utonia has positive opportunity mass on
    `picture` (`B=0.0800`) but also larger false-defer danger (`A=0.1612`);
    Concerto decoder is useful for `door` (`B/A=1.85`) and
    `shower curtain` (`B/A=3.88`) but not for `picture` (`B/A=0.59`);
    PTv3 has favorable `B/A` on dominant classes like `wall` but weak
    recoverability on `picture` (`B/A=0.24`). Predictability is the limiting
    factor. Binary deferral predictors trained on probability/uncertainty/top
    class features achieve PR-AUC around `0.36`-`0.48`, but high-precision
    recall is essentially zero at precision `0.8`/`0.9`/`0.95`. A conservative
    sample-level router therefore leaves the full-FT default almost unchanged
    (`fold0 0.8129 -> 0.8129`, `fold1 0.7924 -> 0.7929` at P80). Reading:
    selective deferral is the right conceptual framing, but logit/probability
    features cannot safely identify deferable points; any further SOTA attempt
    must use feature/region-level evidence or train-split multi-expert decoding,
    not another confidence-based defer rule. Details are in
    `tools/concerto_projection_shortcut/results_cross_model_deferral_scannet20_fullft_default_with_ptv3.md`.
  - A feature-level selective-deferral diagnostic was then run with the same
    full-FT default. The predictor input augments the previous logit/confidence
    features with fixed random projections of raw point features from the
    default and auxiliary experts (Concerto decoder, Sonata, Utonia; PTv3
    remains probability-only in this run). This does not materially solve
    selectability. Feature-mode PR-AUC is comparable to or slightly worse than
    logit-only for most experts (`Concerto decoder` best `0.4733` vs logit
    `0.4778`; `Utonia` best `0.4494` vs logit `0.4457`; `Sonata` best
    `0.3537` vs logit `0.3643`), and high-precision recall remains near zero.
    A sample conservative router moves the full-FT default only marginally
    (`fold0 0.8117 -> 0.8123`, `fold1 0.7926 -> 0.7939` at P80). Reading:
    cross-model complementarity is real, but a shallow pointwise deferral
    selector over projected raw features still cannot safely recover it; a
    serious SOTA attempt would need a train-split feature/region-level fusion
    decoder rather than another pointwise confidence/feature gate. Details are
    in
    `tools/concerto_projection_shortcut/results_cross_model_feature_deferral_scannet20_fullft_default_with_ptv3.md`.
  - The fusion protocol mismatch was also made explicit. The same full-FT
    checkpoint scores `0.7969` in the raw-point aligned single-pass cache
    protocol used by fusion diagnostics, but `0.8075` under the Pointcept
    model_best test path. The 5-expert raw average (`0.8065`) should therefore
    be interpreted as a diagnostic raw-protocol gain, not as an official SOTA
    number. Details are in
    `tools/concerto_projection_shortcut/results_fusion_protocol_alignment.md`.
  - The 5-expert `avgprob_all` row was exported as scene-wise Pointcept-style
    `pred/*.npy` / `submit/*.txt` predictions and re-scored from the saved
    raw-point predictions. This confirms the save/eval path reproduces the
    raw-aligned fusion protocol rather than the Pointcept fragment/voting test
    path: `fullft_single_saved=0.7969`, `avgprob_all_saved=0.8064`.
    Therefore the simple average is a strong raw-protocol baseline but still
    not an official SOTA row by itself. Details are in
    `tools/concerto_projection_shortcut/results_cross_model_fusion_export_scannet20_avgprob5_fullft_ptv3.md`.
  - Region-level expert-choice coherence was tested with Concerto full-FT as
    the default expert. Label-oracle expert choice is spatially coherent at
    fine granularity: point oracle is `0.9004`, region oracle remains high at
    `s4=0.8890`, `s8=0.8842`, `s16=0.8785`, and target-expert region purity is
    `0.9104/0.8724/0.8305` for `s4/s8/s16`. FullFT-default region-defer
    oracles are also well above fullFT single, with the best `s4::Utonia`
    reaching `0.8496`. Reading: complementarity is not purely pointwise noise;
    region-smoothed expert correction is plausible. This is an oracle
    diagnostic only because labels choose region experts. Details are in
    `tools/concerto_projection_shortcut/results_cross_model_region_coherence_scannet20_fullft_default_ptv3.md`.
  - A FullFT-centered residual fusion diagnostic was run after the probability
    and shallow deferral no-gos. This is still a two-fold scene-level
    validation pilot, not a publishable train-split result. A single
    `kl=0.03, safe=2` pilot reaches `0.8097`, and the minimal full-val
    diagnostic with two settings reaches `0.8164` for `kl=0, safe=4`, above
    both `avgprob_all=0.8065` and fullFT single `0.7969` in the same raw
    protocol. The gain comes with weak-class tradeoffs (`picture` drops to
    `0.3999`, `picture->wall` rises to `0.4808`), so this is not a final
    method yet. Reading: unlike hard/selective deferral, residual fusion has
    real SOTA-route signal; the next proper step is a train-split version with
    final val-only evaluation and official-path export. Details are in
    `tools/concerto_projection_shortcut/results_cross_model_residual_fusion_scannet20_fullft_default_with_ptv3_kl003_safe2.md`
    and
    `tools/concerto_projection_shortcut/results_cross_model_residual_fusion_scannet20_fullft_default_with_ptv3_minimal.md`.
  - Train-split residual fusion was launched to convert the val-CV signal into
    a publishable protocol. The new script
    `tools/concerto_projection_shortcut/train_cross_model_residual_fusion_scannet20.py`
    trains on ScanNet train scenes, selects hyperparameters on held-out train
    scenes, and evaluates ScanNet val once. Jobs `135491.qjcm` and
    `135492.qjcm` ran SSL-only expert sets
    (`Concerto fullFT` default + `Concerto decoder`, `Sonata`, `Utonia`) with
    `logit+feature` and `logit-only` inputs, respectively. Both use
    `384` train scenes, `heldout-every=5`, full val, 12 epochs, KL
    `{0,0.03}`, and safe CE `{2,4}`. PTv3 was not included in these two jobs
    because train-split PTv3 raw-prob caches did not exist at launch time.
    PTv3 train-cache export job `135493.qjcm` has now completed for the first
    `384` train scenes under
    `data/runs/ptv3_v151_raw_probs_scannet20/trainval384`; the existing val
    cache was linked into the same cache root. The first PTv3-inclusive
    train-split residual jobs `135494.qjcm` and `135495.qjcm` failed fast
    because the 384-scene export did not cover every ScanNet train scene
    visited by the train loader (`scene0449_02` was missing). Full train-split
    PTv3 export job `135496.qjcm` completed into the same cache root with
    `1201` train scene caches, and the missing scene is now present. The
    PTv3-inclusive residual jobs were relaunched as `135498.qjcm` (feature)
    and `135499.qjcm` (logit). The SSL-only `135491.qjcm`/`135492.qjcm` jobs
    are unaffected.
  - The SSL-only train-split residual fusion rows completed:
    `tools/concerto_projection_shortcut/results_cross_model_residual_fusion_scannet20_train_split_logit_ssl.md`
    and
    `tools/concerto_projection_shortcut/results_cross_model_residual_fusion_scannet20_train_split_feature_ssl.md`.
    In this publishable protocol, the val-CV residual signal does not transfer
    to a clear method gain. Logit-only selects `kl=0.03,safe=2` and gives
    `0.7967` mIoU versus `0.7964` for the fullFT default; feature mode selects
    `kl=0.03,safe=4` and gives `0.7960` versus `0.7960` for fullFT. In both
    cases, simple SSL-only `avgprob_all` is stronger (`0.8019` logit,
    `0.8023` feature) but still below the official-path fullFT reference
    (`~0.8075`). Reading: the earlier `0.8164` residual result was a useful
    val-CV diagnostic signal, but the first train-split SSL-only residual
    decoder is not a publishable SOTA method.
  - The PTv3-inclusive train-split residual fusion rows also completed:
    `tools/concerto_projection_shortcut/results_cross_model_residual_fusion_scannet20_train_split_logit_with_ptv3.md`
    and
    `tools/concerto_projection_shortcut/results_cross_model_residual_fusion_scannet20_train_split_feature_with_ptv3.md`.
    Adding PTv3 raises the simple average but still does not turn residual
    fusion into a publishable SOTA row. Logit+PTv3 selects `kl=0,safe=2` and
    gives `0.7990` versus `0.7970` for the fullFT default, while feature+PTv3
    gives `0.7976` versus `0.7967`. The strongest train-split rows are still
    simple 5-expert averages: `0.8068` for logit mode and `0.8061` for feature
    mode, both just below the official-path fullFT reference (`~0.8075`).
    Reading: cross-model complementarity remains real, but the train-split
    residual decoder recovers only a small safe correction. The SOTA route is
    not validated by this bounded residual-fusion pilot; use it as evidence
    that val-CV fusion signals can overstate publishable gains.
  - PTv3 supervised compatibility fix completed. The earlier invalid PTv3 rows
    were not due to missing checkpoint keys (`missing=0/unexpected=0`) but due
    to released Pointcept v1.5.1 protocol differences. Two concrete mismatches
    were confirmed: v1.5.1 uses `cls_mode` and color normalization
    `color / 127.5 - 1`, while the current repo uses `enc_mode` and
    `color / 255`. Using the official v1.5.1 model/transform code path with the
    current `.npy` ScanNet scenes recovers a valid supervised row: clean
    `0.7697`, random keep `0.2` `0.7143`, class-wise keep `0.2` `0.7107`,
    structured keep `0.2` `0.6521`, feature-zero `0.0269`. Details are in
    `tools/concerto_projection_shortcut/results_ptv3_v151_masking_compat_full.md`.
  - Class-wise keep control completed for Concerto decoder, Concerto linear,
    and Sonata linear. Keeping 20% within each GT class gives nearly the same
    overall mIoU as global random keep 20%: Concerto decoder `0.7626` random vs
    `0.7629` class-wise, Concerto linear `0.7602` vs `0.7603`, Sonata linear
    `0.6951` vs `0.6951`. Thus the retained-subset robustness is not mainly due
    to class-composition drift. `picture` remains fragile and still shifts
    toward `wall`, so the core interpretation stays: overall retained-subset
    robustness hides weak-class fragility. Details are in
    `tools/concerto_projection_shortcut/results_masking_classwise_keep.md`.
  - Sonata ScanNet downstream audit completed as the first external SSL
    model-family anchor. The released Sonata backbone and ScanNet linear head
    were merged into
    `data/weights/sonata/sonata_scannet_linear_merged.pth` and evaluated with
    the same ScanNet stage-wise / oracle-actionability protocol used for
    Concerto. Sonata shows the same broad pattern that candidate-set headroom
    is large while ordinary readout remains weak for wall-adjacent classes:
    base oracle-analysis mIoU is `0.7086`, base `picture` IoU is `0.3582`,
    and `picture -> wall` is `0.4783`; oracle top-5 raises mIoU to `0.9670`
    and `picture` IoU to `0.8867`, while oracle graph top-5 raises `picture`
    to `0.9922`. Stage-wise trace shows `picture_vs_wall` is present but not
    solved by the fixed 20-way readout: point-feature balanced accuracy is
    `0.7501`, refit binary probe on linear logits is `0.8155`, but direct
    fixed logit margin is `0.6452`. This supports the ED framing that the
    readout/actionability gap is not a Concerto-only artifact, although the
    exact stage pattern differs from Concerto. Details are in
    `tools/concerto_projection_shortcut/results_sonata_scannet_point_stagewise_trace.md`
    and
    `tools/concerto_projection_shortcut/results_sonata_scannet_oracle_actionability_analysis.md`.
  - Official-like Sonata ScanNet full fine-tuning also completed at
    `data/runs/scannet_semseg_origin/exp/scannet-ft-sonata-e800`. The final
    evaluation reports mIoU/mAcc/allAcc `0.7955 / 0.8649 / 0.9271`, with
    class-wise IoU including `picture 0.3602`, `door 0.7635`,
    `cabinet 0.7704`, `counter 0.7284`, `sink 0.7337`, and
    `otherfurniture 0.6674`. The full-FT pairwise stagewise/oracle trace is now
    also complete: `picture_vs_wall` reaches point/logit/direct balanced
    accuracy `0.6742 / 0.6734 / 0.6485`, while the base oracle-analysis row is
    `mIoU=0.7770`, `picture=0.3508`, and `picture -> wall=0.5478`; oracle
    top-2 / top-5 raise `picture` to `0.6003 / 0.7700`, and oracle graph
    top-5 raises it to `0.9924`. This makes the backbone-moving Sonata row a
    full parity external SSL anchor, not just a separate aggregate full-FT
    check. Details are in
    `tools/concerto_projection_shortcut/results_sonata_scannet_fullft.md`,
    `tools/concerto_projection_shortcut/results_sonata_fullft_point_stagewise_trace/scannet_point_stagewise_trace.md`,
    and
    `tools/concerto_projection_shortcut/results_sonata_fullft_oracle_actionability/oracle_actionability_analysis.md`.
  - PTv3 v1.5.1 supervised ScanNet20 downstream audit completed on the same
    stage-wise / oracle-actionability protocol. This provides the missing
    supervised anchor for the readout-gap claim. PTv3 has a materially cleaner
    base readout than Concerto/Sonata but still leaves substantial oracle
    headroom. On `picture_vs_wall`, PTv3 reaches point-feature balanced
    accuracy `0.9626`, refit-logit balanced accuracy `0.9529`, direct
    pair-margin accuracy `0.8892`, base `picture` IoU `0.4908`, and
    `picture -> wall` `0.2326`, with oracle top-2 / top-5 raising `picture`
    IoU to `0.8785 / 0.9952`. The same pattern holds for `door_vs_wall` and
    `counter_vs_cabinet`. This narrows the interpretation to: the
    readout/actionability gap is not SSL-exclusive, but it is substantially
    worse for the audited SSL rows on ScanNet20 than for the supervised PTv3
    anchor. Details are in
    `tools/concerto_projection_shortcut/results_ptv3_scannet20_point_stagewise_trace.md`,
    `tools/concerto_projection_shortcut/results_ptv3_scannet20_oracle_actionability.md`,
    and
    `tools/concerto_projection_shortcut/results_cross_model_downstream_audit_scannet20.md`.
  - PTv3 supervised ScanNet200 downstream corroboration is also positive. The
    same official v1.5.1 compatibility path shows that the readout gap does
    not disappear when moving to the 200-class label space: `picture_vs_wall`
    has point-feature balanced accuracy `0.9097`, refit-logit balanced
    accuracy `0.9094`, direct pair-margin accuracy `0.8613`, base `picture`
    IoU `0.4624`, `picture -> wall` `0.2667`, top-2 hit rate `0.8461`, and
    oracle top-5 `picture` IoU `0.8388`. `door_vs_wall` and
    `counter_vs_cabinet` remain strong as well. This is weaker than the
    ScanNet20 supervised anchor but still supports the same reading: there is
    actionable pairwise information beyond the fixed multiclass readout, even
    in a valid supervised baseline on a harder label space. Details are in
    `tools/concerto_projection_shortcut/results_ptv3_scannet200_point_stagewise_trace.md`,
    `tools/concerto_projection_shortcut/results_ptv3_scannet200_oracle_actionability.md`,
    and
    `tools/concerto_projection_shortcut/results_ptv3_scannet200_downstream_audit.md`.
  - PTv3 v1.5.1 supervised masking audit was extended to ScanNet200 and S3DIS
    as dataset-level externality checks. The v1.5.1 compatibility evaluator is
    now generic over segment key, class names, focus class, and confusion
    class. Official PTv3 checkpoints were downloaded from
    `Pointcept/PointTransformerV3` and evaluated through the same retained-voxel
    masking protocol. ScanNet200 is valid and close to the reported supervised
    PTv3 level: clean mIoU `0.3458` vs README-reported `0.353`; random keep
    `0.2` drops to `0.2618`, class-wise keep `0.2` to `0.2640`, structured
    keep `0.2` to `0.2691`, and feature-zero to `0.0019`. The ScanNet200
    `picture` focus class is fragile (`0.3771 -> 0.2442` under random keep
    `0.2`) and shifts toward `wall` (`0.4595 -> 0.6536`). S3DIS Area-5 is also
    valid enough for audit: clean mIoU `0.7052`, random keep `0.2` `0.4589`,
    class-wise keep `0.2` `0.4542`, structured keep `0.2` `0.6393`, and
    feature-zero `0.1138`. This adds cross-dataset evidence that retained-subset
    sparsity robustness is protocol- and dataset-dependent: ScanNet200 and
    S3DIS supervised rows degrade much more under random sparsity than Concerto
    / Sonata ScanNet20 SSL rows, while structured sparsity can be much less
    damaging on S3DIS. ScanNet++ is not yet run because no public compatible
    downstream checkpoint was found in the checked Pointcept/Concerto,
    facebook/sonata, or PointTransformerV3 HF repos. Details are in
    `tools/concerto_projection_shortcut/results_ptv3_scannet200_v151_masking_full.md`
    and
    `tools/concerto_projection_shortcut/results_ptv3_s3dis_v151_masking_full.md`.
  - Full-scene masking scoring completed for available checkpoints only
    (Concerto decoder/linear, Sonata linear, PTv3 ScanNet20/ScanNet200/S3DIS).
    The masking evaluators now optionally write both retained-subset rows and
    `full_nn` rows, where retained logits are propagated back to every original
    voxel by nearest-neighbor assignment before scoring. Random keep `0.2`
    robustness mostly survives this stricter full-scene scoring on ScanNet20:
    Concerto decoder `0.7632` retained vs `0.7527` full, Concerto linear
    `0.7597` vs `0.7519`, Sonata `0.6951` vs `0.6865`, and PTv3 ScanNet20
    `0.7131` vs `0.6995`. In contrast, structured block keep `0.2` collapses
    under full-scene scoring: Concerto decoder `0.7388` retained vs `0.3012`
    full, Concerto linear `0.7409` vs `0.2907`, Sonata `0.6712` vs `0.2662`,
    PTv3 ScanNet20 `0.6573` vs `0.2491`, PTv3 ScanNet200 `0.2425` vs
    `0.0772`, and PTv3 S3DIS `0.6374` vs `0.2881`. This upgrades the masking
    interpretation: retained-subset scoring hides structured missing-region
    damage, random sparsity is still highly redundant on ScanNet20, and the
    effect remains incompatible with a pure coordinate-only explanation because
    feature-zero collapses all rows. Details are in
    `tools/concerto_projection_shortcut/results_masking_fullscene_scoring.md`.
  - Object-style masked-model keep20 extension completed for the same available
    checkpoints. This uses whole-object/stuff masking on raw scenes before the
    normal voxel transform, then scores with the same `full_nn` protocol. It is
    much harsher than random keep20 at similar observed keep fractions: Concerto
    decoder drops `0.7869 -> 0.1982`, Concerto linear `0.7695 -> 0.1990`,
    Sonata linear `0.7167 -> 0.1724`, PTv3 ScanNet20 `0.7699 -> 0.1324`,
    PTv3 ScanNet200 `0.3447 -> 0.0441`, and PTv3 S3DIS `0.7040 -> 0.1083`.
    The weak focus classes collapse with it (`picture` on ScanNet/ScanNet200,
    `board` on S3DIS). This sharpens the masking interpretation: random keep20
    is a weak sparsity regime because scene outline / partial geometry remain,
    while object-style masking and fixed-point stress are much stronger tests of
    missing-support robustness. Details are in
    `tools/concerto_projection_shortcut/results_masking_maskedmodel_full.md`.
  - Masking example export was added to pin down what the new stress regimes
    actually look like on voxelized inputs. For 5 example scenes each from
    ScanNet, ScanNet200, and S3DIS, the exporter now writes
    `clean_voxel`, `random_keep0p2`, `random_keep0p1`, `fixed_points_4000`,
    and `masked_model_keep0p2` under
    `data/runs/masking_examples/<dataset>/<scene>/<condition>/`. The key
    interpretation is that `random_keep0p2` is still a fairly dense regime:
    roughly `18k-33k` points on ScanNet/ScanNet200 and `32k-98k` on S3DIS,
    whereas `fixed_points_4000` is the first regime that reliably drops the
    voxelized input to about `0.8%-4.3%`. The object-style masked-model
    condition removes whole-instance silhouettes and therefore is a distinct
    stress family rather than a point-budget-matched version of keep20. See
    `tools/concerto_projection_shortcut/results_masking_examples.md`.
  - XYZ-MLP PCA RASA pilot completed for the Concerto origin decoder features.
    The experiment trains an xyz-only MLP on ScanNet20 labels, compresses its
    penultimate hidden representation to a 2D PCA target, and linearly predicts
    that task-conditioned coordinate target from frozen Concerto decoder
    features. On a class-capped train sample (`1.2M` points) and a full-val
    reservoir sample (`2.0M` points across all `312` val scenes), the 2D target
    is clearly linearly visible in Concerto features (`val R2=0.4925`; dim0
    `0.5827`, dim1 `0.4070`). However, this factor is not a clean removable
    harmful subspace: `picture_all` R2 is negative (`-0.3470`), projection
    energy is higher for picture points than average (`0.0421` vs `0.0250`),
    and rank-2 RASA-style removal/add-back does not improve the refit
    downstream classifier (`0.7317` mIoU baseline refit, best removal/add-back
    below that). Current interpretation: task-conditioned coordinate-derived
    factors are present, but they are entangled with useful layout/readout
    information; raw subspace removal is diagnostic, not a positive method.
    Details are in
    `tools/concerto_projection_shortcut/results_xyz_mlp_pca_rasa_reservoir.md`.
  - XYZ-MLP PCA error-conditioned energy follow-up completed on the full
    ScanNet val split (`35.9M` points). Hard-class errors to majority classes
    have higher coordinate-factor projection energy than correct hard-class
    points (`0.0234` vs `0.0095`) and slightly higher R2 (`0.3101` vs
    `0.2646`), so the task-conditioned coordinate factor is not completely
    unrelated to hard errors. However, correct majority-class points remain the
    strongest coordinate-factor subset (`energy=0.0279`, `R2=0.5192`), and the
    key `picture_to_wall` subset has high energy (`0.0413`) but near/under-zero
    R2 (`-0.0517`). This supports the current reading: the coordinate-derived
    factor exists and is mildly error-associated, but it is not a clean harmful
    shortcut subspace explaining the picture/wall failure. Details are in
    `tools/concerto_projection_shortcut/results_xyz_mlp_pca_error_energy.md`.
  - Data and run outputs should live under repo-local `data/`.
  - Existing ScanNet is used through a symlink, not copied.
  - Do not run the optional fine-tune, e075/e100, or broad posthoc sweeps
    without a new decision criterion or hypothesis.

## Documentation Policy

- `CURRENT_STATUS.md` is the canonical high-level status document.
- When a stage finishes, the corresponding `results_*.md` / `results_*.csv`
  file should be updated and linked here.
- If a stage is still running, its current state is tracked here with the
  primary log path.

## Best Documents To Read First

1. Objective-level conclusion:
   - [results_arkit_full_causal.md](./results_arkit_full_causal.md)
2. Geometry-vs-coordinate stress result:
   - [results_arkit_full_stress_corrected.md](./results_arkit_full_stress_corrected.md)
3. Official checkpoint large-video ARKit/ScanNet causal battery:
   - [results_official_causal_battery.md](./results_official_causal_battery.md)
4. Large-video checkpoint reference / cross-variant evidence:
   - [results_large_video_reference.md](./results_large_video_reference.md)
5. ScanNet continuation proxy:
   - [results_scannet_proxy_lin.md](./results_scannet_proxy_lin.md)
6. ProjRes v1 gate:
   - [results_projres_v1.md](./results_projres_v1.md)
7. Frozen post-training nuisance surgery:
   - [results_posthoc_nuisance_surgery.md](./results_posthoc_nuisance_surgery.md)
8. SR-LoRA v5 Phase A:
   - [results_sr_lora_phasea.md](./results_sr_lora_phasea.md)
9. DINO patch-bias Step A':
   - [results_dino_patch_bias_stepA.md](./results_dino_patch_bias_stepA.md)
10. Concerto 3D patch-separation Step A:
   - [results_concerto3d_patch_separation_stepA.md](./results_concerto3d_patch_separation_stepA.md)
11. Concerto 3D / DINO exact-patch controls:
   - [results_concerto3d_dino_exact_controls_stepA.md](./results_concerto3d_dino_exact_controls_stepA.md)
12. Concerto 3D stage-wise patch trace:
   - [results_concerto3d_stagewise_trace_stepA.md](./results_concerto3d_stagewise_trace_stepA.md)
13. ScanNet point-level stage-wise trace:
   - [results_scannet_point_stagewise_trace.md](./results_scannet_point_stagewise_trace.md)
14. ScanNet decoder-probe per-class check:
   - [results_scannet_decoder_probe_origin.md](./results_scannet_decoder_probe_origin.md)
15. ScanNet origin decoder stage-wise trace:
   - [results_scannet_decoder_probe_origin_stagewise.md](./results_scannet_decoder_probe_origin_stagewise.md)
16. Confusion-graph residual readout pilot:
   - [results_confusion_residual_readout.md](./results_confusion_residual_readout.md)
17. Top-K pairwise reranking decoder:
   - [results_topk_pairwise_rerank_decoder.md](./results_topk_pairwise_rerank_decoder.md)
18. Oracle/actionability analysis:
   - [results_oracle_actionability_analysis.md](./results_oracle_actionability_analysis.md)
19. Constrained Top-K set decoder:
   - [results_constrained_topk_set_decoder.md](./results_constrained_topk_set_decoder.md)
20. CoDA decoder adapter:
   - [results_coda_decoder_adapter.md](./results_coda_decoder_adapter.md)
21. CoDA transfer-failure analysis:
   - [results_coda_transfer_failure_analysis.md](./results_coda_transfer_failure_analysis.md)
22. CIDA in-loop decoder adaptation:
   - [results_cida_inloop_decoder_adaptation.md](./results_cida_inloop_decoder_adaptation.md)
23. Origin plain LoRA and decoder-capacity-matched LoRA controls:
   - [results_scannet_lora_origin_perclass.md](./results_scannet_lora_origin_perclass.md)
   - [results_scannet_dec_lora_origin_perclass.md](./results_scannet_dec_lora_origin_perclass.md)
24. Official Concerto LoRA recipe check:
   - [results_official_lora_recipe_check.md](./results_official_lora_recipe_check.md)
25. Retrieval/prototype readout:
   - [results_prototype_readout.md](./results_prototype_readout.md)
   - [results_knn_readout_small.md](./results_knn_readout_small.md)
26. LP-FT / class-safe LoRA follow-up:
   - [results_scannet_lora_lpft_classsafe.md](./results_scannet_lora_lpft_classsafe.md)
27. Decoupled classifier readout:
   - [results_decoupled_classifier_readout.md](./results_decoupled_classifier_readout.md)
28. Latent-subgroup decoder:
   - [results_latent_subgroup_decoder.md](./results_latent_subgroup_decoder.md)
29. Region / superpoint diagnostic:
   - [results_region_superpoint_analysis.md](./results_region_superpoint_analysis.md)
30. Purity-aware hybrid region decoder gate:
   - [results_purity_aware_region_readout.md](./results_purity_aware_region_readout.md)
   - [results_purity_aware_region_readout_lowgate.md](./results_purity_aware_region_readout_lowgate.md)
31. Proposal-first / mask-lite region gates:
   - [results_proposal_recall_analysis.md](./results_proposal_recall_analysis.md)
   - [results_proposal_verify_decoder.md](./results_proposal_verify_decoder.md)
32. Masking ranking battery:
   - [results_masking_ranking_battery.md](./results_masking_ranking_battery.md)
   - [results_masking_downloaded_comparators.md](./results_masking_downloaded_comparators.md)
   - [results_masking_classwise_keep.md](./results_masking_classwise_keep.md)
   - [results_ptv3_v151_masking_compat_full.md](./results_ptv3_v151_masking_compat_full.md)
   - [results_masking_battery_full.md](./results_masking_battery_full.md)
   - [results_masking_linear_origin_full.md](./results_masking_linear_origin_full.md)
   - [results_masking_coord_baselines_full.md](./results_masking_coord_baselines_full.md)
   - [results_masking_coord_baselines_balanced_full.md](./results_masking_coord_baselines_balanced_full.md)
   - [results_masking_battery_smoke.md](./results_masking_battery_smoke.md)
33. Sonata ScanNet external SSL audit:
   - [results_sonata_scannet_point_stagewise_trace.md](./results_sonata_scannet_point_stagewise_trace.md)
   - [results_sonata_scannet_oracle_actionability_analysis.md](./results_sonata_scannet_oracle_actionability_analysis.md)
34. PTv3 ScanNet20 supervised downstream audit:
   - [results_ptv3_scannet20_point_stagewise_trace.md](./results_ptv3_scannet20_point_stagewise_trace.md)
   - [results_ptv3_scannet20_oracle_actionability.md](./results_ptv3_scannet20_oracle_actionability.md)
   - [results_cross_model_downstream_audit_scannet20.md](./results_cross_model_downstream_audit_scannet20.md)
35. PTv3 ScanNet200 supervised downstream corroboration:
   - [results_ptv3_scannet200_point_stagewise_trace.md](./results_ptv3_scannet200_point_stagewise_trace.md)
   - [results_ptv3_scannet200_oracle_actionability.md](./results_ptv3_scannet200_oracle_actionability.md)
   - [results_ptv3_scannet200_downstream_audit.md](./results_ptv3_scannet200_downstream_audit.md)
36. ScanNet200 / S3DIS supervised masking externality:
   - [results_ptv3_scannet200_v151_masking_full.md](./results_ptv3_scannet200_v151_masking_full.md)
   - [results_ptv3_s3dis_v151_masking_full.md](./results_ptv3_s3dis_v151_masking_full.md)
37. Full-scene masking scoring:
   - [results_masking_fullscene_scoring.md](./results_masking_fullscene_scoring.md)
   - [results_masking_fullscene_concerto_decoder.md](./results_masking_fullscene_concerto_decoder.md)
   - [results_masking_fullscene_concerto_linear.md](./results_masking_fullscene_concerto_linear.md)
   - [results_masking_fullscene_sonata_linear.md](./results_masking_fullscene_sonata_linear.md)
   - [results_masking_fullscene_ptv3_scannet20.md](./results_masking_fullscene_ptv3_scannet20.md)
   - [results_masking_fullscene_ptv3_scannet200.md](./results_masking_fullscene_ptv3_scannet200.md)
   - [results_masking_fullscene_ptv3_s3dis.md](./results_masking_fullscene_ptv3_s3dis.md)
38. Masking example export / stress interpretation:
   - [results_masking_examples.md](./results_masking_examples.md)
39. XYZ-MLP PCA RASA diagnostic:
   - [results_xyz_mlp_pca_rasa_reservoir.md](./results_xyz_mlp_pca_rasa_reservoir.md)
40. XYZ-MLP PCA error-conditioned energy:
   - [results_xyz_mlp_pca_error_energy.md](./results_xyz_mlp_pca_error_energy.md)
41. Coordinate projection residual handoff:
   - [HANDOFF_PROJRES_V1.md](./HANDOFF_PROJRES_V1.md)
42. Short narrative summary:
   - [results_interim_summary_2026-04-06.md](./results_interim_summary_2026-04-06.md)
43. Reproduction / runner overview:
   - [README.md](./README.md)

## Official Large-Video Checkpoint Causal Battery

Source:
- [results_official_causal_battery.md](./results_official_causal_battery.md)

Setup:
- Weight: `data/weights/concerto/pretrain-concerto-v1m1-2-large-video.pth`.
- Scope: released full pretraining checkpoint for the large-video variant.
  These numbers remain useful as cross-variant evidence, but the main Concerto
  paper variant now needs the frozen-backbone head-refit diagnostic below.
- ARKit root: `data/arkitscenes_absmeta`.
- ScanNet root: `data/concerto_scannet_imagepoint_absmeta`.
- Smoke size: 32 batches per dataset and mode.

Key numbers:

| dataset | mode | enc2d loss | delta vs baseline |
| --- | --- | ---: | ---: |
| ARKit val | baseline | 2.737827 | 0.000000 |
| ARKit val | global target permutation | 3.324298 | +0.586471 |
| ARKit val | cross-image target swap | 3.371361 | +0.633534 |
| ARKit val | cross-scene target swap | 3.324546 | +0.586719 |
| ScanNet val | baseline | 3.416545 | 0.000000 |
| ScanNet val | global target permutation | 5.466386 | +2.049841 |
| ScanNet val | cross-image target swap | 5.502719 | +2.086174 |
| ScanNet val | cross-scene target swap | 5.467065 | +2.050520 |

Interpretation:
- The released large-video full pretraining checkpoint remains sensitive to
  target swaps on ARKit validation.
- The same signal is stronger on Concerto-preprocessed ScanNet validation.
- This supports the read that the issue is not only an ARKit-continued artifact.
- `concerto_base_origin.pth` is the Concerto paper main-variant backbone weight
  but does not contain enc2d / patch projection heads. The main-variant line
  therefore uses frozen-backbone head-refit rather than exact unreleased enc2d
  head recovery.
- We do **not** rerun Concerto main pretraining from scratch for Step 0/0.5.
  The goal of this line is to audit the **released** main-variant artifact, not
  to study a new reproduction with a changed recipe. Using the released
  backbone and refitting only the missing enc2d / patch-projection head keeps
  the object of study anchored to the official checkpoint while avoiding a
  large six-dataset pretraining confound. A scratch rerun would change both the
  optimization path and the artifact under audit, so it is reserved for a
  different question.

## Main-Variant Step 0/0.5 Plan

Current implementation:
- Data prep:
  - `setup_concerto_six_imagepoint.sh`
  - `prepare_concerto_imagepoint_splits.py`
  - `verify_concerto_six_datasets.py`
  - `download_hf_dataset_shard.py`
  - `submit_hf_dataset_download_shard_abciq_qc.sh`
  - `launch_structured3d_parallel_download.sh`
- Main-variant diagnostics:
  - `fit_main_variant_enc2d_head.py`
  - `fit_main_variant_coord_mlp_rival.py`
  - `run_main_variant_step05.sh`
  - `submit_main_variant_step05_abciq_qf.sh`

Dataset scope:
- ScanNet, ScanNet++, Structured3D, S3DIS, ARKitScenes, HM3D.
- RE10K is excluded from this mainline because it belongs to the large-video
  variant.

Current data state:
- Ready: ARKitScenes, ScanNet, ScanNet++, S3DIS, HM3D, and Structured3D
  Concerto image-point absmeta roots under repo-local `data/`.
- The final six-dataset Step 0/0.5 job `133050.qjcm` finished with
  `Exit_status=0`, `rt_QF=1`, tag `main-origin-six-step05`.
- A dedicated six-dataset **eval-only** rerun `134116.qjcm` finished with
  `Exit_status=0` and wrote the paper-ready all-six causal battery to
  `data/runs/main_variant_enc2d_headfit/main-origin-six-step05-all6eval-rerun2/`.
- Target-corruption distance job `133090.qjcm` finished with `Exit_status=0`.

Acceptance:
- Main-variant causal battery:
  `data/runs/main_variant_enc2d_headfit/main-origin-six-step05-all6eval-rerun2/results_main_variant_causal_battery.md`.
  - Six-dataset deltas are positive on all indoor datasets, but with materially
    different magnitudes:
    - `arkit`: `+0.94 / +1.04 / +1.07`
    - `scannet`: `+1.65 / +1.47 / +1.85`
    - `scannetpp`: `+2.19 / +2.20 / +2.41`
    - `s3dis`: `+0.30 / +0.39 / +0.22`
    - `hm3d`: `+1.72 / +1.92 / +2.03`
    - `structured3d`: `+1.83 / +1.81 / +2.12`
  - Reading: the target-swap sensitivity is not confined to ARKit/ScanNet.
    It is strongest on `scannetpp`, `hm3d`, and `structured3d`, moderate on
    `scannet`, present but much weaker on `s3dis`, and consistently positive
    across `global_target_permutation`, `cross_image_target_swap`, and
    `cross_scene_target_swap`.
  - Main-variant coord-MLP rival:
    `data/runs/main_variant_coord_mlp_rival/main-origin-six-step05/results_official_coord_mlp_rival.md`.
  - Paper-facing six-dataset calibration:
    `tools/concerto_projection_shortcut/results_coord_mlp_rival_six_dataset_calibration.md`.
    Relative to the mean of global permutation, cross-image swap, and
    cross-scene swap, closure is `15.1% / 34.4% / 41.4% / -73.0% / 45.6% /
    23.5%` for `arkit / scannet / scannetpp / s3dis / hm3d / structured3d`.
    Mean closure is `14.5%`, so do not write that a coord-only rival explains
    the six-dataset objective response on average.
  - S3DIS high-val corrected calibration:
    `tools/concerto_projection_shortcut/results_coord_mlp_rival_six_dataset_calibration_s3dis_highval.md`.
    The S3DIS-only high-validation rerun writes
    `data/runs/main_variant_coord_mlp_rival/s3dis-highval-probe/results_official_coord_mlp_rival.md`
    with `107235` train rows and `29822` val rows, replacing the original
    `449`-row val cache. The S3DIS coord loss becomes `6.1365`, relative
    position `0.8668`, and closure `13.3%`. With this corrected S3DIS row,
    positive closure holds on `6/6` datasets and mean closure is `28.9%`.
    Reading: the original S3DIS negative was primarily a validation-extraction
    artifact, but S3DIS is still the weakest coordinate-only rival dataset and
    should be described explicitly rather than hidden in an average.
  - S3DIS failure diagnostic:
    `tools/concerto_projection_shortcut/results_coord_mlp_s3dis_failure_diagnostic.md`.
    S3DIS is the outlier for three concrete reasons: the coord-rival val cache
    has only `449` rows versus `15k-28k` rows for the other datasets; its
    clean-to-mean-corruption denominator is small (`0.3003`); and train/val
    target statistics shift much more strongly (`target mean cosine=0.8536`,
    coordinate mean shift `0.4489`). A S3DIS-only coord MLP reduces train loss
    to `5.8708` but still gives val loss `6.2347`, worse than the mean
    corruption reference (`6.1764`). Reading: S3DIS is not evidence for a
    clean no-coordinate result; it is a sample/shift-sensitive outlier and
    should be reported separately or excluded from any coordinate-closure
    average.
  - Follow-up `135489.qjcm` completed:
    `tools/concerto_projection_shortcut/submit_s3dis_coord_mlp_highval_abciq_qf.sh`.
    This reran S3DIS-only coord extraction with `max_val_batches=512` and
    `max_rows_per_batch=4096` without overwriting the canonical repo-level
    coord-rival CSV. It confirms the `449` val rows were an extraction-cap
    artifact, not a true lack of valid S3DIS image-point rows.
- Main-variant target-corruption distance:
  `data/runs/main_variant_target_corruption_distance/main-origin-six-step05/results_target_corruption_distance.md`.
- SR-LoRA Phase A:
  [results_sr_lora_phasea.md](./results_sr_lora_phasea.md) and
  [results_sr_lora_phasea.csv](./results_sr_lora_phasea.csv).
  - Smoke: `133093.qjcm`, `sr-lora-v5-smoke3-r4-d03`.
  - Pilot: `133095.qjcm`, `sr-lora-v5-pilot-r4-d03-i256-qf4`.
  - Matrix: `133097.qjcm` to `133100.qjcm`.
  - Corrected `m=0.1` follow-up: `133102.qjcm`.
  - `m=0.2` pilot/follow-up: `133104.qjcm`, `133105.qjcm`.
  - Loss-based hinge smoke: `133137.qjcm`.
  - Failed first follow-up: `133101.qjcm`, base-config/large-checkpoint shape
    mismatch; corrected by large ScanNet proxy config.
  - Decision: coord-only similarity-margin SR-LoRA v5 is training-stable but
    downstream no-go. Loss-based hinge is available for the next minimal
    ablation.
  - ScanNet class-wise diagnosis is complete. Baseline weak classes are led by
    `picture` (`0.3962` IoU, mostly confused as `wall`), followed by
    `counter`, `desk`, `sink`, `otherfurniture`, `cabinet`,
    `shower curtain`, and `door`. SR-LoRA m=0.1/m=0.2 only gives narrow gains
    on `sink` and `shower curtain`; it does not solve `picture -> wall` and
    m=0.2 worsens `picture`. See
    `tools/concerto_projection_shortcut/results_scannet_classwise_diagnosis.md`.
  - Next SR-style work should be class-aware or confusion-aware, not another
    global coordinate-rival similarity-margin sweep.
  - ScanNet counterfactual downstream stress is complete. Constant
    floor-relative `z_shift` and post-center `xy_shift` barely move mIoU
    (`|delta| <= 0.0008`), while `z_scale_050` drops mIoU by `0.0120`.
    This suggests downstream dependence is more about relative vertical
    scale/shape than absolute coordinate offset. See
    `tools/concerto_projection_shortcut/results_scannet_counterfactual_downstream.md`.
  - DINO patch-bias Step A' is complete on Concerto-preprocessed ScanNet
    images. DINOv2 patch features separate `picture` vs `wall` patches
    reasonably well and expose patch position very strongly: `raw_dino`
    picture/wall balanced accuracy `0.7797`, AUC `0.8827`, position R2 mean
    `0.8787`. A minimal rank-2 RASA-lite position removal barely reduces
    position R2 (`0.8651`) and does not hurt picture/wall separation. This
    supports a teacher-side positional-bias diagnostic, but not a DINO-only
    failure claim; the 2D teacher carries semantics and position together. The
    raw A' patch set is not the same subset used by the first-pass Concerto A''
    rows, so DINO-vs-Concerto semantic-transfer comparisons should use the
    exact-patch stage-wise trace below, not the raw `0.7797` number. See
    `tools/concerto_projection_shortcut/results_dino_patch_bias_stepA.md`.
  - Concerto 3D patch-separation Step A is complete on the same
    Concerto-preprocessed ScanNet image-point data, using
    `pretrain-concerto-v1m1-2-large-video.pth`. Job `133151.qjcm` exposed that
    `segment20.npy` was not loaded by `DefaultImagePointDataset`; the runner now
    creates lightweight `segment.npy -> segment20.npy` aliases. Corrected job
    `133152.qjcm` finished with `Exit_status=0`, walltime `00:04:51`.
    - `encoder_pooled`: picture/wall balanced accuracy `0.5381`, AUC `0.6662`.
    - `patch_proj`: picture/wall balanced accuracy `0.5547`, AUC `0.6980`.
    - First-pass interpretation was intentionally downgraded after the
      exact-patch controls below. This result is useful as a warning signal, but
      the DINO Step A' `raw_dino` comparison used a different patch subset and
      must not be used as direct evidence that DINO semantics are broadly lost
      in 3D alignment. See
      `tools/concerto_projection_shortcut/results_concerto3d_patch_separation_stepA.md`.
  - Concerto 3D / DINO exact-patch stage-wise trace is complete. Rerun job
    `133188.qjcm` finished with `Exit_status=0`, time use `00:21:05`.
    - Controls include the exact same patch set for DINO and Concerto,
      unweighted / balanced / positive-weighted probes, 100-iteration
      class-stratified bootstrap CIs, multiple confused class pairs, and two
      extra downstream-aligned stages: pooled ScanNet linear-probe backbone
      features and pooled 20-way linear logits.
    - `picture_vs_wall`, balanced probe:
      `dino_exact` `0.6146` CI `[0.5564, 0.6671]`, `encoder_pooled` `0.5548`
      CI `[0.4884, 0.6127]`, `patch_proj` `0.5359` CI `[0.4706, 0.6088]`,
      `linear_feat_pooled` `0.5653`, and `linear_logits_pooled` `0.7128`.
    - Other balanced-probe examples:
      `desk_vs_wall`: DINO `0.6389`, encoder `0.6399`, patch-proj `0.5711`,
      pooled logits `0.8106`; `desk_vs_table`: DINO `0.8495`, encoder
      `0.8692`, patch-proj `0.8828`, pooled logits `0.9060`;
      `counter_vs_cabinet`: all stages are weak-to-moderate (`0.4454` to
      `0.5569`); `door_vs_wall`: patch stages are near chance to weak, pooled
      logits `0.5557`.
    - `shower_curtain_vs_wall` is skipped in this patch-level trace because no
      validation `shower curtain` patch survived the image-point purity filter.
    - Interpretation: the data do not support a broad "DINO semantics are lost
      in 3D alignment" claim. The bottleneck is pair/stage/subset-specific. See
      `tools/concerto_projection_shortcut/results_concerto3d_stagewise_trace_stepA.md`.
  - ScanNet point-level stage-wise trace is complete. Job `133187.qjcm`
    finished with `Exit_status=0`, time use `00:04:13`.
    - This uses the official ScanNet linear-probe data path and checkpoint,
      capped at 60k rows per target class, with the same weak class pairs and
      bootstrap protocol.
    - Balanced point-feature separability is already high for most weak pairs:
      `picture_vs_wall` `0.7041`, `counter_vs_cabinet` `0.9276`,
      `desk_vs_wall` `0.9264`, `desk_vs_table` `0.8711`,
      `sink_vs_cabinet` `0.9707`, `door_vs_wall` `0.8662`, and
      `shower_curtain_vs_wall` `0.9801`.
    - Linear-logit probes improve or preserve these separations:
      `picture_vs_wall` `0.7602`, `desk_vs_wall` `0.9476`,
      `door_vs_wall` `0.9517`, and `desk_vs_table` `0.9101`.
    - The fixed 20-way readout still predicts many `picture` points as `wall`:
      `picture -> wall` fraction `0.5969`, `picture -> picture` fraction
      `0.3968`. This confirms the class-wise bottleneck remains at the final
      readout even though pairwise point features are separable.
    - Interpretation: standard ScanNet point features are not semantically
      collapsed for these pairs. The remaining failure is a readout /
      calibration / class-prior issue for `picture`, not a generic frozen
      encoder semantic-transfer loss. See
      `tools/concerto_projection_shortcut/results_scannet_point_stagewise_trace.md`.

## ARKit Full Causal Branch

Source:
- [results_arkit_full_causal.md](./results_arkit_full_causal.md)

Key numbers:

| experiment | enc2d last | delta vs baseline |
| --- | ---: | ---: |
| baseline | 5.9470 | 0.0000 |
| coord_mlp | 6.4204 | +0.4734 |
| global_target_permutation | 6.4563 | +0.5093 |
| cross_scene_target_swap | 6.4526 | +0.5056 |
| cross_image_target_swap | 6.4873 | +0.5403 |
| coord_residual_target | 7.2936 | +1.3466 |

Interpretation:
- `coord_mlp` remains surprisingly competitive.
- Strong correspondence corruption hurts, but not catastrophically.
- This supports a `correspondence-induced coordinate shortcut` /
  `scene-coordinate cache shortcut` reading.
- The current `coord_residual_target` implementation is not yet a successful fix.

## Corrected Stress Test

Source:
- [results_arkit_full_stress_corrected.md](./results_arkit_full_stress_corrected.md)

Key numbers:

| checkpoint | clean | local_surface_destroy | z_flip | xy_swap | roll_90_x |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.453537 | 3.897785 | 3.980001 | 3.450245 | 3.845464 |
| coord_mlp | 3.536469 | 3.544549 | 3.620394 | 3.537728 | 3.601248 |
| coord_residual_target | 3.628110 | 4.223210 | 4.181930 | 3.624020 | 4.302245 |

Interpretation:
- The original branch is more sensitive to geometry destruction and scene-frame
  transforms.
- `coord_mlp` is much flatter under these perturbations.
- This is consistent with weaker dependence on local geometry.

## ScanNet Downstream Status

Status:
- The original / coord_mlp / no-enc2d / no-enc2d-renorm continuation proxy is
  finished enough for the current decision.
- The safest readout is "downstream effect is real but limited."
- The next gate is not another critique arm; it is whether the projection
  residual fix can match or beat original Concerto on ScanNet linear.

What was confirmed:
- The dataset path and weights are usable.
- A single-GPU safe smoke run reaches actual training:
  - historical path: `exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log`
- The official ScanNet linear gate now completes on the safe single-GPU path:
  - historical path: `exp/concerto/scannet-proxy-official-origin-lin/train.log`
  - final `Val result: mIoU/mAcc/allAcc = 0.1752 / 0.2467 / 0.6167`
  - checkpoints:
    - `exp/concerto/scannet-proxy-official-origin-lin/model/model_best.pth`
    - `exp/concerto/scannet-proxy-official-origin-lin/model/model_last.pth`

Previous failure mode:
- The original 2-GPU official gate crashed repeatedly in the distributed spawn path:
  - historical path: `tools/concerto_projection_shortcut/logs/scannet_gate.launch.log`
- The previous `no-enc2d-renorm` full post-train test aborted while writing
  `.npy` outputs because of local disk pressure. Validation metrics are still
  usable.

Current interpretation:
- The blocker is not the basic dataset path.
- The old blocker was the multi-GPU `mp.spawn` path on this machine.
- For current work, use the validation-only ScanNet linear config to avoid the
  full-test disk failure path.
- `projres_v1b` shows that full coordinate removal was too blunt; partial
  target residualization around `beta=0.75` recovers meaningful downstream
  performance, but still does not beat original.
- `projres_v1c` shows that swapping to lower-capacity / height-biased static
  priors does not close the gap; `mlp_z` is best within v1c but remains below
  the v1b best.
- The long-horizon e025 same-stage check shows v1b can match the original
  downstream proxy much more closely than the 5-epoch gate suggested:
  - same-stage original: 0.5531 / 0.5531 mIoU
  - v1b `combo-b075-a001`: 0.5526 / 0.5526 mIoU
  - this is an effective tie, not a +0.01 fix-and-beat-original result
- Frozen post-training nuisance surgery on the original e025 checkpoint is also
  near-tie:
  - SPLICE-3D `height+xyz`: 0.5509 / 0.5509 mIoU
  - SPLICE-3D `height`: 0.5510 / 0.5510 mIoU
  - HLNS channel-group proxy `height+xyz`: 0.5515 / 0.5515 mIoU
  - Residual Recycling `height+xyz`, `coord9`: 0.5506 / 0.5506 mIoU
  - this is not a win, but it is a cheap post-training path that leaves the
    backbone frozen and does not materially damage the ScanNet proxy.
- The e025 posthoc stress downstream gate is no-go:
  - SPLICE-3D height max stress gain: `+0.0027` on `z_flip`
  - SPLICE-3D height+xyz max stress gain: `+0.0029` on `z_flip`
  - Residual Recycling coord9 max stress gain: `+0.0023` on `z_flip`
  - pass threshold was `clean >= original - 0.005` and any stress
    `>= original + 0.005`; clean passes, stress does not.
- The Step 1 local-geometry smoke is no-go for the current `geom_local9`
  descriptor:
  - offline frozen-cache original baseline: `0.49433` mIoU
  - `concat=[x, phi]`: `0.49435`, delta `+0.00002`
  - `residual_expert_global=W0x+Aphi`: `0.49427`, delta `-0.00006`
  - pass threshold was `+0.003`; this does not support launching UGNR with the
    tested local geometry descriptor.
- On ABCI-Q, keep using the validated `torchrun` / `pbsdsh` path described in
  [HANDOFF_PROJRES_V1.md](./HANDOFF_PROJRES_V1.md).

## Active Downstream Jobs

Running now:
- None observed at the time of this update.

Recently completed:
- `132837.qjcm`: Step 1 local-geometry smoke fit-only rerun after robust
  float64 ridge solve, ABCI-Q `rt_QF=1`, completed.
  - result root:
    `data/runs/step1_geometry_smoke/arkit-full-original-long-e025-qf32-continue/geom_smoke`
  - result: `concat` delta `+0.00002`, `residual_expert_global` delta
    `-0.00006`; no-go by the `+0.003` geometry-signal criterion.
- `132835.qjcm`: Step 1 local-geometry main cache extraction, ABCI-Q
  `rt_QF=1`, created the 524288 train / 131072 val row `geom_local9` cache but
  hit a singular float32 normal-equation solve in the fit stage.
- `132832.qjcm` and `132833.qjcm`: Step 1 tiny path smoke and calibration,
  ABCI-Q `rt_QF=1`, both completed.
- `132789.qjcm`: Residual Recycling e025 stress downstream, ABCI-Q `rt_QF=1`,
  completed.
  - result root:
    `data/runs/posthoc_stress_e025pilot_recycle`
  - result: clean `0.5509`, max stress gain `+0.0023` on `z_flip`; no-go by
    the `+0.005` stress criterion.
- `132780.qjcm`: Residual Recycling e025 posthoc linear probe, ABCI-Q
  `rt_QF=1`, completed.
  - result root:
    `data/runs/posthoc_surgery_e025pilot/original-long-e025-qf32/recycle_height_xyz_coord9_g1.0_r1.0`
  - result: `0.5506 / 0.5506` ScanNet proxy mIoU.
- `132775.qjcm`: SPLICE-3D e025 stress downstream, ABCI-Q `rt_QF=1`,
  completed.
  - result root:
    `data/runs/posthoc_stress_e025pilot`
  - result: clean preserved, max stress gains below `+0.005`; no-go for Stage 1.
- `132600.qjcm` and `132602.qjcm`: attempted fresh e050 same-stage
  continuations on ABCI-Q `rt_QF=8`, both failed by walltime.
  - `132600.qjcm`: original e050, `Exit_status=-29`, stopped at epoch 6.
  - `132602.qjcm`: v1b e050, `Exit_status=-29`, stopped at epoch 6.
  - No valid e050 same-stage checkpoint should be read from these runs.
- `132601.qjcm` and `132603.qjcm`: dependent follow-ups for the failed e050
  attempts did not produce valid e050 follow-up results.
- `132608.qjcm`: frozen post-training nuisance surgery e025 pilot, ABCI-Q
  `rt_QF=1`, completed.
  - result root:
    `data/runs/posthoc_surgery_e025pilot/original-long-e025-qf32`
  - results:
    SPLICE-3D `height+xyz` 0.5509 / 0.5509, SPLICE-3D `height` 0.5510 /
    0.5510, HLNS `height+xyz` 0.5515 / 0.5515 ScanNet proxy mIoU.
- `132455.qjcm` and `132457.qjcm`: long-horizon e025 continuations on ABCI-Q
  `rt_QF=8`, both `Exit_status=0`.
  - `132455.qjcm`: original reference, walltime `03:11:21`
  - `132457.qjcm`: v1b `combo-b075-a001`, walltime `03:11:46`
  - requested walltime was `03:50:00`, so the setting was sufficient without
    excessive slack.
  - checkpoints:
    `exp/concerto/arkit-full-original-long-e025-qf32-continue/model/model_last.pth`
    and
    `exp/concerto/arkit-full-projres-v1b-combo-b075-a001-long-e025-qf32-continue/model/model_last.pth`
- `132456.qjcm` and `132458.qjcm`: dependent ScanNet proxy follow-ups on
  ABCI-Q `rt_QF=1`, both `Exit_status=0`.
  - walltimes: `00:50:15` and `00:49:57`
  - requested walltime was `01:05:00`
  - results:
    original e025 `0.5531 / 0.5531`, v1b e025 `0.5526 / 0.5526`
  - summary root:
    `data/runs/projres_long/summaries/long-e025-qf32`
- `132277.qjcm`: ProjRes v1c z-prior fit on ABCI-Q `rt_QF=1`,
  `Exit_status=0`.
  - cache reused:
    `data/runs/projres_v1/priors/cache`
  - fitted priors:
    `linear_z`, `mlp_z`
  - selected by cosine loss:
    `mlp_z`
- `132278.qjcm` to `132283.qjcm`: ProjRes v1c 6-arm prior-family smoke matrix
  on ABCI-Q `rt_QF=1`.
  - summary root:
    `data/runs/projres_v1c/summaries/h10016-qf1-v1c-prior256`
  - logs reached 190 to 193 steps before the 35 minute walltime; partial smoke
    summaries were generated with a 128-step minimum.
  - selected top arms:
    `linz-b075-a000`, `mlpz-b075-a001`, `linxyz-b075-a001`
- `132284.qjcm` to `132286.qjcm`: ProjRes v1c 5-epoch continuations, each on
  ABCI-Q `rt_QF=4` (4 nodes / 16 H100 GPUs), all `Exit_status=0`.
  - concurrent allocation: 12 nodes / 48 H100 GPUs
  - walltimes: about 47 minutes
  - checkpoints:
    `exp/concerto/arkit-full-projres-v1c-linz-b075-a000-h10016x3-qf16-v1c-continue/model/model_last.pth`,
    `exp/concerto/arkit-full-projres-v1c-mlpz-b075-a001-h10016x3-qf16-v1c-continue/model/model_last.pth`,
    `exp/concerto/arkit-full-projres-v1c-linxyz-b075-a001-h10016x3-qf16-v1c-continue/model/model_last.pth`
- `132287.qjcm` to `132289.qjcm`: ProjRes v1c follow-up stress + ScanNet
  linear gates on ABCI-Q `rt_QF=1`, all `Exit_status=0`.
  - summary root:
    `data/runs/projres_v1c/summaries/h10016x3-qf16-v1c`
  - result: no strong-go for all three arms
- `132208.qjcm`: ProjRes v1b metric sanity on ABCI-Q `rt_QF=1`,
  `Exit_status=0`.
  - setting: v1a-equivalent `beta=1.0`, `alpha=0.05`, 16 train steps
  - result: `coord_projection_loss_check=0.0`
- `132209.qjcm` to `132219.qjcm`: ProjRes v1b 11-arm smoke matrix on ABCI-Q
  `rt_QF=1`, all `Exit_status=0`.
  - summary root:
    `data/runs/projres_v1b/summaries/h10016-qf1-v1b-pre256`
  - selected top arms:
    `combo-b075-a001`, `penalty-b000-a002`, `resonly-b075-a000`,
    `combo-b050-a002`
- `132220.qjcm` to `132223.qjcm`: ProjRes v1b 5-epoch continuations, each on
  ABCI-Q `rt_QF=4` (4 nodes / 16 H100 GPUs), all `Exit_status=0`.
  - concurrent allocation: 16 nodes / 64 H100 GPUs
  - walltimes: about 47 minutes
  - checkpoints:
    `exp/concerto/arkit-full-projres-v1b-combo-b075-a001-h10016x4-qf16-continue/model/model_last.pth`,
    `exp/concerto/arkit-full-projres-v1b-penalty-b000-a002-h10016x4-qf16-continue/model/model_last.pth`,
    `exp/concerto/arkit-full-projres-v1b-resonly-b075-a000-h10016x4-qf16-continue/model/model_last.pth`,
    `exp/concerto/arkit-full-projres-v1b-combo-b050-a002-h10016x4-qf16-continue/model/model_last.pth`
- `132255.qjcm` to `132258.qjcm`: ProjRes v1b follow-up stress + ScanNet
  linear gates on ABCI-Q `rt_QF=1`, all `Exit_status=0`.
  - summary root:
    `data/runs/projres_v1b/summaries/h10016x4-qf16`
  - result: no strong-go for all four arms
- `132196.qjcm`: ProjRes v1 5-epoch continuation on ABCI-Q `rt_QF=8`
  (8 nodes / 32 H100 GPUs), `Exit_status=0`, walltime `00:39:37`.
  - experiment:
    `arkit-full-projres-v1a-alpha005-h10032-qf32-continue`
  - checkpoint:
    `exp/concerto/arkit-full-projres-v1a-alpha005-h10032-qf32-continue/model/model_last.pth`
  - main log:
    `data/logs/abciq/projres_v1_continue_qf16_132196.qjcm.log`
  - rank logs:
    `data/runs/projres_v1/logs/multinode/132196.qjcm_arkit-full-projres-v1a-alpha005-h10032-qf32-continue_20260415_233258/logs/`
- `132198.qjcm`: ProjRes v1 follow-up on ABCI-Q `rt_QF=1`,
  `Exit_status=0`, walltime `00:50:06`.
  - script:
    `tools/concerto_projection_shortcut/submit_projres_v1_followup_abciq_qf.sh`
  - log:
    `data/logs/abciq/projres_v1_followup_132198.qjcm.log`
  - outputs:
    `data/runs/projres_v1/summaries/h10032-qf32/arkit-full-projres-v1a-alpha005-h10032-qf32-continue_stress.csv`
    and
    `data/runs/projres_v1/summaries/h10032-qf32/scannet-proxy-projres-v1a-alpha005-h10032-qf32-lin_gate.json`

ProjRes v1 gate result:
- selected alpha: `0.05`
- final continuation metrics:
  - `loss=8.0899`, `enc2d_loss=8.0655`,
    `coord_residual_enc2d_loss=6.3309`,
    `coord_alignment_loss=0.0200`,
    `coord_pred_energy=0.0020`, `coord_residual_norm=0.7308`
- ARKit stress, enc2d loss mean over 20 batches:
  - clean `8.022868`
  - local surface destroy `9.115467`
  - z flip `8.941781`
  - xy swap `8.022733`
  - roll 90 x `9.353544`
- ScanNet linear gate:
  - ProjRes v1a last/best mIoU: `0.3627` / `0.3627`
  - original continuation last/best mIoU: `0.4794` / `0.4552`
  - no-enc2d-renorm last/best mIoU: `0.3794` / `0.3802`
  - deltas vs original: `-0.1167` last, `-0.0925` best
  - deltas vs no-enc2d-renorm: `-0.0167` last, `-0.0175` best
  - decision: `strong_go=false`, `linear_gate_not_strong_go`
- Summary:
  - [results_projres_v1.md](./results_projres_v1.md)

ProjRes v1b gate result:
- best arm: `combo-b075-a001`
- beta / alpha: `0.75` / `0.01`
- final continuation metrics:
  - `loss=7.8700`, `enc2d_loss=7.6640`,
    `coord_residual_enc2d_loss=7.2912`,
    `coord_alignment_loss=1.7203`,
    `coord_removed_energy=0.0734`,
    `coord_pred_energy=0.1720`, `coord_residual_norm=0.8921`,
    `coord_projection_loss_check=0.0000`
- ARKit stress, enc2d loss mean over 20 batches:
  - clean `7.649344`
  - local surface destroy `8.862813`
  - z flip `8.860726`
  - xy swap `7.673076`
  - roll 90 x `9.093753`
- ScanNet linear gate:
  - best v1b last/best mIoU: `0.4220` / `0.4220`
  - original continuation last/best mIoU: `0.4794` / `0.4552`
  - no-enc2d-renorm last/best mIoU: `0.3794` / `0.3802`
  - deltas vs original: `-0.0574` last, `-0.0332` best
  - deltas vs no-enc2d-renorm: `+0.0426` last, `+0.0418` best
  - decision: `strong_go=false`, `linear_gate_not_strong_go`
- Summary:
  - [results_projres_v1.md](./results_projres_v1.md)

ProjRes v1c gate result:
- hypothesis:
  - keep `beta=0.75` and test lower-capacity / height-biased priors instead of
    widening the beta/alpha grid.
- fitted z-priors:
  - `linear_z`: cosine loss `0.735445`, target energy `0.080087`, residual norm
    `0.958630`
  - `mlp_z`: cosine loss `0.643186`, target energy `0.136156`, residual norm
    `0.928692`
- continued arms:
  - `linz-b075-a000`: `linear_z`, `beta=0.75`, `alpha=0.00`
  - `mlpz-b075-a001`: `mlp_z`, `beta=0.75`, `alpha=0.01`
  - `linxyz-b075-a001`: `linear_xyz`, `beta=0.75`, `alpha=0.01`
- best v1c arm:
  - `mlpz-b075-a001`
- best v1c final continuation metrics:
  - `loss=7.8765`, `enc2d_loss=7.6774`,
    `coord_residual_enc2d_loss=7.2917`,
    `coord_alignment_loss=1.6903`,
    `coord_removed_energy=0.0736`,
    `coord_pred_energy=0.1690`, `coord_residual_norm=0.8903`,
    `coord_projection_loss_check=0.0000`
- best v1c ScanNet linear gate:
  - last/best mIoU: `0.4186` / `0.4186`
  - deltas vs original: `-0.0608` last, `-0.0366` best
  - deltas vs no-enc2d-renorm: `+0.0392` last, `+0.0384` best
  - decision: `strong_go=false`, `linear_gate_not_strong_go`
- Summary:
  - [results_projres_v1.md](./results_projres_v1.md)

Latest ABCI-Q launcher status, 2026-04-16 JST:
- The useful hint from
  `/groups/qgah50055/ide/3d-sans-3dscans/Pointcept/configs/*.sh` is that
  ABCI-Q Pointcept jobs use `python -m torch.distributed.run` with
  `tools/ddp_train.py`, not the original Pointcept `mp.spawn` launcher.
- This checkout now has the same launcher option:
  - [tools/ddp_train.py](../../tools/ddp_train.py)
  - [scripts/train.sh](../../scripts/train.sh) with
    `POINTCEPT_TRAIN_LAUNCHER=torchrun`
- A lightweight batch-only diagnostic was added for ARKit DDP checks:
  - [debug_arkit_ddp_batches.py](./debug_arkit_ddp_batches.py)
  - [submit_debug_arkit_ddp_batches_abciq_qf.sh](./submit_debug_arkit_ddp_batches_abciq_qf.sh)
- Batch-only diagnostic job `132175.qjcm` completed successfully:
  - walltime: `00:01:12`
  - result: `Exit_status = 0`
  - log: `data/logs/abciq/debug_arkit_ddp_batches_132175.qjcm.log`
  - result: all ranks completed 16 batches through DataLoader, CUDA copy, and
    all-reduce. The current stall is therefore not reproduced by DataLoader
    alone.
- `POINTCEPT_TRACE_STEPS=1` now prints per-rank full-training step trace around
  `next(DataLoader)`, `run_step`, `forward`, `backward`, and `after_step`.
- Full-training trace results:
  - `132176.qjcm`: 8-step `alpha=0.05` run completed with `Exit_status = 0`
    in `00:02:09`.
  - `132177.qjcm`: 16-step run was stopped at `00:03:48`; all ranks fetched
    the 9th batch (`iter=8`) and reached `before_run_step`, but no rank reached
    `after_run_step`.
  - `132178.qjcm`: run-step trace was stopped at `00:02:54`; all ranks reached
    `run_step_before_forward` on the first iteration, rank 2 reached
    `run_step_after_forward` / `run_step_before_backward`, and ranks 0/1/3 did
    not return from forward. This established the pre-fix failure as a
    model-forward rank divergence/hang, not a pure DataLoader issue.
- On ABCI-Q H100, the current stable single-node setting is:
  - `POINTCEPT_TRAIN_LAUNCHER=torchrun`
  - `NCCL_STABLE_MODE=1`
  - `NCCL_P2P_DISABLE=1`
  - `NCCL_NET_GDR_LEVEL=0`
  - `CONCERTO_NUM_WORKER=1`
- A 4-GPU torchrun smoke with this stable NCCL mode completed:
  - job: `132168.qjcm`
  - walltime: `00:01:44`
  - result: `Exit_status = 0`
  - log: `data/logs/abciq/projres_v1_smoke_qf1_132168.qjcm.log`
  - output:
    `data/runs/projres_v1/summaries/h10016-qf4gtorchstable/selected_smoke.json`
- The earlier longer-smoke stalls were isolated to the distributed metric
  reduction at the end of `Concerto.forward`, not to DataLoader, optimizer, or
  H100 memory pressure:
  - pre-fix stopped/stalled jobs: `132169`, `132170`, `132172`, `132173`,
    `132184`, `132185`, `132187`, `132189`
  - symptom: one rank returned from forward while other ranks were still inside
    forward-side collectives.
  - fix: use a fixed distributed result key order, fill missing coord metric
    scalars with zero, reduce detached metric copies, and keep
    `loss_for_backward` separate from reduced logging metrics.
  - this also removes the PyTorch `c10d::allreduce_` autograd warning.
- Post-fix 4-GPU H100 smoke results:
  - `132190.qjcm`: 16 steps, flash on, `Exit_status = 0`, walltime `00:02:27`.
  - `132191.qjcm`: 64 steps, flash on, `Exit_status = 0`, walltime `00:04:57`.
  - `132192.qjcm`: 16 steps after detached metric reduction, `Exit_status = 0`,
    walltime `00:02:24`, no `c10d::allreduce_` autograd warning.
- Multi-node continuation validation:
  - `132194.qjcm`: 4 nodes / 16 H100 GPUs, short continuation validation,
    `Exit_status = 0`, walltime `00:03:58`.
  - `132195.qjcm`: 8 nodes / 32 H100 GPUs, 1 epoch x 16-step continuation
    validation, `Exit_status = 0`, walltime `00:02:08`.
  - The H100 continuation config now accepts `CONCERTO_EPOCH` for bounded
    validation jobs, and the `pbsdsh` launcher explicitly forwards
    `CONCERTO_*` env values to every node.
- Current smoke-only selected alpha artifact:
  - `data/runs/projres_v1/summaries/h10016-qf1fixed64/selected_smoke.json`
  - selected `alpha=0.05`
  - This is a 64-step single-node smoke artifact. It has now been validated by
    short 4-node and 8-node continuation runs.

Completed setup jobs:
- ABCI-Q env setup job `132080.qjcm` completed with `Exit_status = 0`.
  - log: `data/logs/abciq/env_setup_132080.qjcm.log`
  - result: created `data/venv/pointcept-concerto-py311-cu124` and validated
    `torch`, `torch_scatter`, `spconv`, `pointops`, `transformers`, and
    `pointcept` imports on an `rt_QF=1` GPU allocation.
- ABCI-Q dry-run job `132093.qjcm` completed successfully.
  - log: `data/logs/abciq/projres_v1_132093.qjcm.log`
  - result: GPU preflight passed and `DRY_RUN=1` printed only repo-local
    `data/...` paths.

Prepared data:
- `data/scannet` is a symlink to
  `/groups/qgah50055/ide/3d-sans-3dscans/scannet`.
- ARKit compressed snapshot exists under `data/concerto_arkitscenes_compressed`.
- ARKit extracted data exists under `data/arkitscenes`.
- ARKit absolute metadata exists under `data/arkitscenes_absmeta`.
- DINOv2 cache exists under `data/hf-home`.
- Concerto official weights exist under `data/weights/concerto`.

Stopped job:
- The local `projres_v1` chain was stopped during Stage 1 prior cache
  extraction.
- Log:
  - `/mnt/urashima/users/minesawa/concerto_shortcut_runs/projres_v1/logs/run_projres_v1_chain_20260415_131727_setsid.log`
- It produced logs only. No selected prior, smoke checkpoint, continuation
  checkpoint, stress csv, or ScanNet linear result was produced.

Expected next stage:
1. Finish the active e050 same-stage check for original and
   `projres_v1b combo-b075-a001`.
2. Do not launch the optional fine-tune from either arm under the current gate.
3. Do not automatically extend to e075/e100 until the e050 ScanNet follow-up is
   read out. If e050 is still a near tie or better with improved shortcut
   metrics, then stage the next checkpoint using the fixed target-epoch /
   stop-epoch resume path.
4. Keep the posthoc nuisance-surgery line as the cheap fallback while the
   expensive continuation jobs run. The e025 pilot shows frozen post-training
   stays within `0.0022` mIoU of the original e025 reference.
5. Keep the fixed DDP metric reduction and ABCI-Q `torchrun` path; those
   infrastructure changes are validated.

## Useful Logs And Artifacts

- ARKit causal summary:
  - [results_arkit_full_causal.csv](./results_arkit_full_causal.csv)
  - [results_arkit_full_causal.md](./results_arkit_full_causal.md)
- ARKit stress summary:
  - [results_arkit_full_stress_corrected.csv](./results_arkit_full_stress_corrected.csv)
  - [results_arkit_full_stress_corrected.md](./results_arkit_full_stress_corrected.md)
- Interim write-up:
  - [results_interim_summary_2026-04-06.md](./results_interim_summary_2026-04-06.md)
- ScanNet gate crash log:
  - historical path: `tools/concerto_projection_shortcut/logs/scannet_gate.launch.log`
- ScanNet gate success log:
  - historical path: `exp/concerto/scannet-proxy-official-origin-lin/train.log`
- ProjRes v1 result:
  - [results_projres_v1.md](./results_projres_v1.md)
- ScanNet gate result note:
  - [results_scannet_gate_2026-04-09.md](./results_scannet_gate_2026-04-09.md)
- ScanNet safe smoke log:
  - historical path: `exp/concerto/scannet-proxy-official-origin-lin-safe-smoke/train.log`
- Pipeline status:
  - [scannet_pipeline_status.md](./scannet_pipeline_status.md)
- Origin plain LoRA per-class control:
  - [results_scannet_lora_origin_perclass.md](./results_scannet_lora_origin_perclass.md)
- Origin decoder-capacity-matched LoRA control:
  - [results_scannet_dec_lora_origin_perclass.md](./results_scannet_dec_lora_origin_perclass.md)
- Official Concerto LoRA recipe check:
  - [results_official_lora_recipe_check.md](./results_official_lora_recipe_check.md)
- Projection residual handoff:
  - [HANDOFF_PROJRES_V1.md](./HANDOFF_PROJRES_V1.md)

## Immediate Next Step

## 2026-04-26 Object-Level / Binding-Profile Gap Closure

Generated paper-facing framework artifacts:

- `results_recoverability_rrec_max.md` / `.csv`: `R_rec^max` table for
  Concerto, Sonata, Utonia, and PTv3. The main recovery suite is now fixed to
  six pre-specified recovery families: class-prior correction / decoupled
  classifier, prototype-or-kNN readout, constrained Top-K reranking,
  fixed-rank LoRA, LP-FT warm-start adaptation, and full fine-tuning.
  `R_rec` is protocol-matched by base representation/readout. In particular,
  LP-FT belongs to the Concerto linear-head family and is not mixed into the
  Concerto decoder-probe oracle denominator. CoDA/CIDA/region/proposal/subgroup
  variants are appendix exploratory rows rather than part of the main
  `R_rec^max` suite.
  `results_recoverability_fixed_suite_methods.md` / `.csv` records the
  family-level rows. The external-model fixed recovery suites are still mostly
  pending, so the recovery-suite claim remains Concerto-centric while external
  models support oracle/actionability comparisons.
- Follow-up protocol-matched recovery jobs submitted on `abciq`:
  - `135520.qjcm`: Concerto linear-head oracle/actionability denominator for
    the LP-FT/linear-head recovery row.
  - `135524.qjcm`: LP-FT plain linear-head checkpoint through the same
    oracle/actionability evaluator, to make the LP-FT numerator as
    protocol-matched as possible.
  - `135521.qjcm`: Sonata linear class-prior / decoupled-classifier frozen
    recovery.
  - `135522.qjcm`: Sonata linear prototype/kNN frozen recovery.
  - `135523.qjcm`: Sonata linear constrained Top-K frozen recovery.
  Utonia and PTv3 are not sent through these generic Pointcept recovery scripts:
  Utonia uses a custom released-stack inference path, and PTv3 is a supervised
  control with a separate compatibility evaluator. Add protocol-specific
  recovery only if the base/readout/evaluator path can be matched cleanly.
- Completed recovery follow-ups from this batch:
  - Concerto linear-head oracle/actionability denominator (`135520.qjcm`):
    base mIoU `0.7615`, oracle top-2 `0.9171`, oracle top-5 `0.9839`;
    picture base `0.4014`, oracle top-2 `0.8013`, top-5 `0.9394`.
  - LP-FT plain through the same evaluator (`135524.qjcm`): base mIoU
    `0.7780`, picture `0.4139`; oracle top-2 remains high at `0.9263`
    mIoU and `0.8198` picture. Linear-head adaptation recovery is therefore
    nonzero but still far from closing the oracle headroom.
  - Sonata class-prior / decoupled classifier (`135521.qjcm`): best aggregate
    row `tau0p25_bias` gives mIoU `0.7107` (`+0.0017`) but picture drops
    (`0.3506`, `-0.0073`). Best safe picture row is essentially flat.
  - Sonata constrained Top-K (`135523.qjcm`): base remains the best mIoU row
    (`0.7091`); best picture row gives only `+0.0006` picture and slightly
    lower mIoU. This mirrors the Concerto result: candidate-set reranking
    does not recover the oracle headroom.
  - Sonata prototype/kNN (`135522.qjcm`) is still running.
- `results_binding_profile_summary.png` / `.pdf`: central binding-profile
  heatmap generated from `results_binding_profile_summary.csv`.
- `results_binding_profile_summary_panels.png` / `.pdf`: paper-facing panel
  version separating train-side counterfactuals, actionability headroom, and
  support-stress damage instead of forcing heterogeneous quantities into a
  single scalar.
- `3D-NEPA/results/pointgpt_object_pretext_summary.md` / `.csv`: object pretext
  summary now includes the official masked checkpoint as a checkpoint-only row
  with no local pretraining log, plus no-mask and no-mask order-random loss
  trajectories.
- `3D-NEPA/results/ptgpt_shapenetpart_support_stress_paper_table.md` / `.csv`:
  ShapeNetPart support-stress paper table. Official and no-mask PointGPT-S rows
  are close on clean class-avg IoU (`0.8335` vs `0.8287`), while random keep20
  is weaker than structured keep20 and semantic part removal is the strongest
  stress in both rows.

New long-running object jobs submitted:

- `135471.qjcm`: PointGPT-S `mask_ratio=0.7` + random token order pretraining
  (`cfgs/PointGPT-S/pretrain_orderrandom.yaml`, 24h QF). This is the missing
  mask-on order-random row needed to separate masking and causal-order effects.
- `135482.qjcm`: dependent PointGPT-S `mask_ratio=0.7` + random token order
  ScanObjectNN `obj_bg` fine-tune, queued with `afterok:135471.qjcm`.
- `135483.qjcm`: dependent readout/support-stress audit for the same row,
  queued with `afterok:135482.qjcm`.
- `135472.qjcm`, `135474.qjcm`, `135476.qjcm`, `135478.qjcm`, `135480.qjcm`,
  `135481.qjcm`: ScanObjectNN `obj_bg` seed repeats for official masked,
  no-mask, and no-mask order-random rows, seeds 1/2.
- `135473.qjcm`, `135475.qjcm`, `135477.qjcm`, `135479.qjcm`: ShapeNetPart
  seed repeats for official masked and no-mask rows, seeds 1/2.
- `135484.qjcm`: ShapeNetPart support-stress audit first attempt failed
  immediately because the eval script lacked the PointGPT root on
  `PYTHONPATH` for `pointnet2_ops`.
- `135485.qjcm`: ShapeNetPart support-stress audit rerun after adding
  `PYTHONPATH=${POINTGPT_DIR}`. It then failed at checkpoint loading because
  PyTorch 2.6+ defaults to `weights_only=True`.
- `135486.qjcm`: ShapeNetPart support-stress audit rerun after setting
  `torch.load(..., weights_only=False)` for trusted local ShapeNetPart
  checkpoints. It then failed because the eval script read `[B, 50, N]` logits
  as `[B, N, 50]`.
- `135487.qjcm`: ShapeNetPart support-stress audit rerun after adding an output
  shape guard/transposition. It still failed due to NumPy advanced-indexing axis
  behavior when selecting category-specific part logits.
- `135488.qjcm`: ShapeNetPart support-stress audit rerun after replacing
  advanced indexing with `np.take(..., axis=1)`. This adds
  random/structured keep20, `xyz_zero`, and two part-aware stress variants:
  `part_drop_largest` and
  `part_keep20_per_part`.

Completed ShapeNetPart support-stress result:

- Official PointGPT-S ShapeNetPart: clean class-avg IoU `0.8335`, random
  keep80/50/20/10 `0.8226 / 0.8184 / 0.6929 / 0.5443`, structured
  keep80/50/20/10 `0.8200 / 0.7674 / 0.6454 / 0.6093`,
  part-drop-largest `0.4844`, part-keep20-per-part `0.6927`, xyz-zero
  `0.2588`.
- No-mask PointGPT-S ShapeNetPart: clean class-avg IoU `0.8287`, random
  keep80/50/20/10 `0.8204 / 0.8131 / 0.7016 / 0.5518`, structured
  keep80/50/20/10 `0.8126 / 0.7673 / 0.6546 / 0.6471`,
  part-drop-largest `0.4711`, part-keep20-per-part `0.6992`, xyz-zero
  `0.2327`.
- Interpretation: dense part transfer is not classification-only. No-mask stays
  close to official on clean ShapeNetPart, and both rows show that part-aware
  removal (`part_drop_largest`) is a much stronger stress than random keep20 /
  per-part keep20.

Completed PointGPT-S ScanObjectNN support-stress severity curves:

- Official `obj_bg`: clean `0.9105`, random keep80/50/20/10
  `0.9139 / 0.8812 / 0.4337 / 0.2616`, structured keep80/50/20/10
  `0.8881 / 0.7453 / 0.2100 / 0.1239`, xyz-zero `0.0929`.
- No-mask `obj_bg`: clean `0.8916`, random keep80/50/20/10
  `0.8726 / 0.8726 / 0.5112 / 0.2392`, structured keep80/50/20/10
  `0.8606 / 0.6936 / 0.2306 / 0.1205`, xyz-zero `0.0929`.
- No-mask + order-random `obj_bg`: clean `0.8744`, random keep80/50/20/10
  `0.8881 / 0.8640 / 0.4234 / 0.1997`, structured keep80/50/20/10
  `0.8537 / 0.6781 / 0.2151 / 0.1687`, xyz-zero `0.0723`.
- Mask-on + order-random `obj_bg`: clean `0.8847`, random keep80/50/20/10
  `0.8744 / 0.8589 / 0.5800 / 0.2719`, structured keep80/50/20/10
  `0.8537 / 0.7005 / 0.2513 / 0.1652`, xyz-zero `0.0723`.

Completed mask-on order-random readout/support audit:

- Readout top-1/top-2/top-5: `0.8795 / 0.9552 / 0.9983`.
- Hardest pair: `bed -> sofa`; direct pair top-1 `0.9219`; binary-probe
  balanced accuracy `0.8972`.
- Support-stress clean/random keep20/structured keep20/xyz-zero:
  `0.8916 / 0.6007 / 0.2754 / 0.0723`.
- Interpretation: mask-on random token order is not collapse-inducing, but it
  remains a meaningful perturbation relative to mask removal alone. This
  supports the scoped 2x2 claim that masking is weakly binding while causal
  order is more binding but not catastrophic.

Initial log check:

- The pretrain job loads `pretrain_orderrandom.yaml` with `mask_ratio: 0.7` and
  `order_mode: random`; data and DDP initialization are healthy.
- ScanObjectNN seed jobs load the intended checkpoints and set the requested
  seeds.
- ShapeNetPart seed jobs start training with the intended checkpoints and seed
  argument. `segmentation/main.py` now accepts `--seed`, and
  `scripts/local/pointgpt_s_shapenetpart_ft.sh` forwards it.

Remaining after these complete:

- Summarize seed variance if the repeat jobs complete. If repeats fail or
  exceed walltime, report only the available point estimate and avoid
  seed-variance claims.
- ShapeNetPart support-stress rows are already folded into
  `results_binding_profile_summary.md`; still copy the short interpretation
  into the object-level paper section when drafting.

1. Treat the current pair-emphasis decoder-family pilots (CoDA/CIDA) as no-go.
   The useful signal is diagnostic: moving `picture` away from `wall` is
   possible, but the current losses damage the full weak-class decision surface.
2. Treat plain origin LoRA as a matched-head positive control only in the
   linear-head family: it improves over same-head no-LoRA and reduces
   `picture -> wall`.
3. The decoder-capacity-matched LoRA control is complete and is not positive
   against the decoder baseline. Do not launch weak-class or class-safety
   variants of this exact decoder+LoRA setup without a new hypothesis. The
   official Concerto LoRA recipe does differ materially, but it requires the
   main-variant large pretraining checkpoint. Obtain that checkpoint before
   claiming or testing official LoRA reproduction; otherwise treat the current
   local base-origin LoRA lines as diagnostic rather than a positive method
   path.
4. Treat retrieval/prototype, decoupled classifier, and latent-subgroup readout
   as completed offline readout-family negatives. The latent-subgroup
   diagnostic is still useful: it supports class-internal subgroup shift for
   `picture`, but sub-center readout does not produce a paper-relevant positive.
   Do not continue offline readout tweaks without changing the protocol or
   obtaining a stronger validation criterion.
5. Treat region/superpoint as diagnostic-positive but not yet a positive
   method. Fine region top-k oracles preserve headroom, but region-logit
   smoothing hurts and coarse regions merge `picture` into `wall`. A serious
   region method should use object-mask-quality regions / learned proposals,
   not coarse voxel averaging.
6. Treat PHRD zero-train as no-go. Label-free purity proxies are too generic:
   they detect confident/pure regions overall, but for `picture` they often
   identify confidently wall-dominated regions. Do not continue thresholded
   point/region logit mixing without a class-specific learned mask/proposal
   mechanism.
7. Treat proposal-first as diagnostic-positive but current PVD fusion as no-go.
   Fine `s4` proposals have enough high-purity recall for `picture`, but
   base-pred connected components collapse and scalar proposal logit boosts
   reduce `picture -> wall` while lowering `picture` IoU. Do not continue this
   exact PVD boost family. If region-family work continues, change the protocol
   to proposal-native classification or learned point-mask assignment.
8. Treat the masking/ranking pilot as useful and now supervised-comparator
   backed, but still not shortcut-proof by itself. The merged released Sonata
   linear checkpoint is a valid external SSL comparator and shows the same
   qualitative sparsity tolerance/feature-zero collapse pattern. The class-wise
   keep control shows that random-mask class-composition drift is not the main
   driver of high retained-subset mIoU. The downloaded PTv3 supervised/PPT rows
   under the current repo/data protocol remain invalid, but a v1.5.1-compatible
   evaluation path for the released PTv3 supervised checkpoint now recovers a
   valid supervised row (`0.7697` clean, `0.7143` random keep `0.2`, `0.6521`
   structured keep `0.2`). This introduces a real ranking signal: Concerto
   decoder/linear are more retained-sparsity robust than supervised PTv3 in this
   protocol, while all methods collapse under feature-zero and drop much more
   under object-style masked-model keep20 than under random keep20. Do not
   overclaim it as coordinate-shortcut proof; use it as retained-subset
   redundancy / weak-class fragility evidence, with masked-model keep20 and
   fixed-points as the stronger stress conditions.
9. Keep the completed CIDA artifacts and use only batch-size-1 val numbers from
   `*-eval-b1` runs for reporting. Keep the origin LoRA classwise outputs under
   `data/runs/scannet_lora_origin/classwise/`.
10. Keep monitoring through ABCI-compatible `qstat` when jobs are active:
   - `qstat | awk -v u="$USER" 'NR==1 || NR==2 || $0 ~ u {print}'`
11. Keep the current completed artifacts:
   - `data/runs/projres_v1/summaries/h10032-qf32`
   - `data/runs/projres_v1b/summaries/h10016-qf1-v1b-pre256`
   - `data/runs/projres_v1b/summaries/h10016x4-qf16`
   - `data/runs/projres_v1c/summaries/h10016x3-qf16-v1c`
   - `data/runs/projres_long/summaries/long-e025-qf32`
   - `data/runs/posthoc_surgery_e025pilot`
   - `data/runs/cida_inloop_decoder_adaptation/cida-base-i1200-eval-b1`
   - `data/runs/cida_inloop_decoder_adaptation/cida-strong-i1200-eval-b1`
   - `data/runs/scannet_dec_lora_origin/classwise`
   - `data/runs/scannet_lora_origin/classwise/`
   - `data/runs/scannet_decoder_probe_origin/latent_subgroup_decoder/`
   - `data/runs/scannet_decoder_probe_origin/region_superpoint_analysis/`
   - `data/runs/scannet_decoder_probe_origin/purity_aware_region_readout/`
   - `data/runs/scannet_decoder_probe_origin/purity_aware_region_readout_lowgate/`
