# When Pretext Bottlenecks Don't Bind: A Mechanistic Audit of 3D Self-Supervised Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
*(Anonymous Repository for NeurIPS Submission)*

This repository contains the official implementation of our proposed **Binding Audit Framework**, a systematic protocol designed to evaluate whether the intended pretext bottlenecks in 3D Self-Supervised Learning (SSL) genuinely drive the learning of downstream semantics, or if they are bypassed via spatial redundancies.

> **TL;DR:** High downstream accuracy in 3D SSL does not certify pretext faithfulness. By auditing scene-level joint-embedding models (Concerto, Sonata, Utonia) and object-level autoregressive models (PointGPT family), we expose two distinct structural risks: **Shortcut-satisfiable objectives** (where correspondence losses contain dataset-dependent coordinate-satisfiable components) and **Slack bottlenecks** (where intended geometric constraints like masking can be relaxed without collapsing downstream accuracy).

---

## 📖 Overview: The Pretext-Downstream Gap

Modern 3D SSL operates on the assumption that solving difficult pretext tasks (e.g., cross-modal correspondence, masked patch prediction) forces the encoder to learn robust semantic representations. 

However, 3D point clouds leak low-order geometric redundancy (absolute XYZ coordinates, local patch topology). We introduce a **Binding Audit** to systematically test if the pretext constraint actually binds the downstream representation. We categorize binding failures into two types:

1. **Shortcut-Satisfiable Objectives (e.g., Concerto):** The pretext objective reacts strongly to target corruption, and a purely geometric alternative (a "blind" coordinate-only MLP) can close a dataset-dependent fraction of that clean-to-corrupted gap. This shows that the objective contains a coordinate-satisfiable component; it does not by itself prove that the released encoder relies only on coordinates.
2. **Slack Bottlenecks (e.g., PointGPT / AR Masking):** The advertised constraint (e.g., severe point masking to prevent positional leakage) can be completely removed during pretraining, yet downstream classification performance remains largely unaffected.

Furthermore, through rigorous **stage-wise and oracle readout decompositions**, we demonstrate that the primary downstream failure in these models is often not complete representation collapse, but rather a **structural calibration failure** where minority classes (e.g., *picture*) are linearly separable in the feature space but are swallowed by massive geometric backgrounds (e.g., *wall*) during standard multiclass linear probing.

---
