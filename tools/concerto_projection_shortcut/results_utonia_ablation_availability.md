# Utonia Ablation Checkpoint Availability

## Local / Public Weight Check

Checked local `data/weights/utonia/` and the public HuggingFace model repository
`Pointcept/Utonia` on `2026-04-25`.

Locally available weights:

- `data/weights/utonia/utonia.pth`
- `data/weights/utonia/pretrain-utonia-v1m1-0-base_stagev2.pth`
- `data/weights/utonia/utonia_linear_prob_head_sc.pth`

Public HuggingFace files:

- `.gitattributes`
- `README.md`
- `pretrain-utonia-v1m1-0-base_stagev2.pth`
- `utonia.pth`
- `utonia_linear_prob_head_sc.pth`

No public ablation checkpoints were found for:

- scale/data ablations such as `38M + 83k`, `137M + 83k`, `137M + all data`;
- design ablations such as `w/o RoPE`, `w/o modality blinding`, or
  `w/o perceptual granularity rescale`.

## Consequence For This Paper

The current audit can use Utonia only as a released-stack constructive
comparator. It shows that the large Concerto/Sonata `picture -> wall`
fixed-readout failure is not inevitable across recent 2D-3D SSL-style released
stacks.

It does **not** identify why Utonia is cleaner. The improvement could be due to
scale, data mixture, cross-domain pretraining, Perceptual Granularity Rescale,
RoPE, Causal Modality Blinding, the released ScanNet head, or their interaction.

Safe phrasing:

> Utonia's released cross-domain stack substantially reduces the audited
> wall-confusion/fixed-margin failure relative to Concerto and Sonata, showing
> that the gap is not universal. We do not attribute this reduction to a single
> Utonia design component without ablation-level checkpoints.

