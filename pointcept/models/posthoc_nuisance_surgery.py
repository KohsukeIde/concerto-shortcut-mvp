"""Post-hoc nuisance surgery modules for Concerto/Pointcept.

This file adds two frozen-feature editors and a drop-in segmentor wrapper:

- Splice3DEditor: global task-preserving nuisance projection.
- HLNSEditor: head-localized (channel-group proxy) nuisance projection.
- PosthocEditedSegmentorV2: loads a pretrained backbone, applies a frozen editor,
  then trains/evaluates the usual linear segmentation head.

The implementation is intentionally lightweight so it can plug into the current
Pointcept/Concerto codebase without changing the backbone or the training loop.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point

from .builder import MODELS, MODULES, build_model


def _unpool_point(point: Point) -> Point:
    """Match DefaultSegmentorV2's feature reconstruction path."""
    if isinstance(point, Point):
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
    return point


@MODULES.register_module()
class LinearFeatureEditorBase(nn.Module):
    """Frozen affine editor: y = ((x - mean) @ W^T) + bias.

    The editor is meant to be fit offline and loaded from a checkpoint.
    """

    def __init__(self, feature_dim: int, preserve_mean: bool = True, **kwargs):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.preserve_mean = bool(preserve_mean)
        self.register_buffer("transform", torch.eye(self.feature_dim))
        self.register_buffer("mean", torch.zeros(self.feature_dim))
        self.register_buffer("bias", torch.zeros(self.feature_dim))
        self.metadata: Dict[str, Any] = {}

    def forward(self, feat: torch.Tensor, point: Optional[Point] = None, input_dict: Optional[dict] = None) -> torch.Tensor:
        x = feat
        if x.shape[-1] != self.transform.shape[0]:
            raise ValueError(
                f"Feature dim mismatch: got {x.shape[-1]}, editor expects {self.transform.shape[0]}"
            )
        x_centered = x - self.mean
        y = x_centered.matmul(self.transform.t()) + self.bias
        if self.preserve_mean:
            y = y + self.mean
        return y

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}, preserve_mean={self.preserve_mean}"


@MODULES.register_module()
class Splice3DEditor(LinearFeatureEditorBase):
    """Task-preserving nuisance projection editor.

    In the fitting script this is implemented as a SPLICE-style approximation that
    removes nuisance directions only after projecting them away from a task basis.
    """

    pass


@MODULES.register_module()
class HLNSEditor(LinearFeatureEditorBase):
    """Head-localized nuisance surgery editor.

    In the current Pointcept linear-probe path only the final point feature tensor
    is exposed. This implementation therefore uses contiguous channel groups as a
    head proxy and stores the resulting block-diagonal transform as a single frozen
    matrix for easy deployment.
    """

    def __init__(self, feature_dim: int, preserve_mean: bool = True, num_groups: int = 16, **kwargs):
        super().__init__(feature_dim=feature_dim, preserve_mean=preserve_mean)
        self.num_groups = int(num_groups)

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, preserve_mean={self.preserve_mean}, "
            f"num_groups={self.num_groups}"
        )


@MODELS.register_module()
class PosthocEditedSegmentorV2(nn.Module):
    """Drop-in replacement for DefaultSegmentorV2 with a frozen post-hoc editor.

    The backbone is loaded from a pretrained checkpoint and frozen. A frozen editor
    is then applied to the final point features before the linear segmentation head.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_out_channels: int,
        backbone: Optional[dict] = None,
        criteria: Optional[list] = None,
        freeze_backbone: bool = True,
        backbone_path: Optional[str] = None,
        keywords: str = "module.student.backbone",
        replacements: str = "module.backbone",
        editor_cfg: Optional[dict] = None,
        editor_path: Optional[str] = None,
        freeze_editor: bool = True,
        return_edit_metrics: bool = False,
    ):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes) if num_classes > 0 else nn.Identity()
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = bool(freeze_backbone)
        self.return_edit_metrics = bool(return_edit_metrics)
        self.keywords = keywords
        self.replacements = replacements

        if self.backbone is None:
            raise ValueError("PosthocEditedSegmentorV2 requires a backbone config.")
        if not backbone_path:
            raise ValueError("PosthocEditedSegmentorV2 requires backbone_path.")

        checkpoint = torch.load(backbone_path, map_location="cpu", weights_only=False)
        self._backbone_load(checkpoint)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if editor_cfg is None:
            raise ValueError("PosthocEditedSegmentorV2 requires editor_cfg.")
        self.editor = MODULES.build(editor_cfg)
        if editor_path:
            self._editor_load(editor_path)
        if freeze_editor:
            for p in self.editor.parameters():
                p.requires_grad = False

    def _backbone_load(self, checkpoint: dict) -> None:
        weight = OrderedDict()
        state_dict = checkpoint.get("state_dict", checkpoint)
        for key, value in state_dict.items():
            if not key.startswith("module."):
                key = "module." + key
            if self.keywords in key:
                key = key.replace(self.keywords, self.replacements)
                key = key[7:]  # module.xxx -> xxx
                if key.startswith("backbone."):
                    key = key[9:]
                weight[key] = value
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        print(f"[PosthocEditedSegmentorV2] backbone missing keys: {load_state_info[0]}")
        print(f"[PosthocEditedSegmentorV2] backbone unexpected keys: {load_state_info[1]}")

    def _editor_load(self, editor_path: str) -> None:
        checkpoint = torch.load(editor_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        metadata = checkpoint.get("metadata", {})
        load_info = self.editor.load_state_dict(state_dict, strict=False)
        if hasattr(self.editor, "metadata"):
            self.editor.metadata = metadata
        print(f"[PosthocEditedSegmentorV2] editor missing keys: {load_info[0]}")
        print(f"[PosthocEditedSegmentorV2] editor unexpected keys: {load_info[1]}")

    def forward(self, input_dict: dict, return_point: bool = False):
        point = Point(input_dict)
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        if isinstance(point, Point):
            point = _unpool_point(point)
            feat = point.feat
        else:
            feat = point

        feat_edited = self.editor(feat, point=point if isinstance(point, Point) else None, input_dict=input_dict)
        if isinstance(point, Point):
            point.feat = feat_edited

        seg_logits = self.seg_head(feat_edited)
        return_dict: Dict[str, Any] = {}
        if return_point and isinstance(point, Point):
            return_dict["point"] = point

        if self.return_edit_metrics:
            with torch.no_grad():
                delta = feat_edited - feat
                denom = feat.pow(2).sum(dim=-1).clamp_min(1e-6)
                edit_energy = delta.pow(2).sum(dim=-1) / denom
                return_dict["edit_energy"] = edit_energy.mean()

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        elif "segment" in input_dict:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict
