import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import torch_cluster
from peft import LoraConfig, get_peft_model
from collections import OrderedDict

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultLORASegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        backbone_path=None,
        keywords=None,
        replacements=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.keywords = keywords
        self.replacements = replacements
        self.backbone = build_model(backbone)
        backbone_weight = torch.load(
            backbone_path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        self.backbone_load(backbone_weight)

        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                # target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(
                self.backbone.enc,
                lora_config,
                autocast_adapter_dtype=False,
            )

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            self.backbone.enc.print_trainable_parameters()

    def backbone_load(self, checkpoint):
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            if self.keywords in key:
                key = key.replace(self.keywords, self.replacements)
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
            if key.startswith("backbone."):
                key = key[9:]
            weight[key] = value
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        print(f"Missing keys: {load_state_info[0]}")
        print(f"Unexpected keys: {load_state_info[1]}")

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.freeze_backbone and not self.use_lora:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassSafeLORASegmentorV2(DefaultLORASegmentorV2):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        backbone_path=None,
        keywords=None,
        replacements=None,
        anchor_path=None,
        anchor_keywords="module",
        anchor_replacements="module",
        weak_classes=(10, 11, 17, 15, 7),
        weak_loss_weight=0.2,
        safe_kl_weight=0.05,
        dist_kl_weight=0.02,
        kl_temperature=2.0,
    ):
        super().__init__(
            num_classes=num_classes,
            backbone_out_channels=backbone_out_channels,
            backbone=backbone,
            criteria=criteria,
            freeze_backbone=freeze_backbone,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            backbone_path=backbone_path,
            keywords=keywords,
            replacements=replacements,
        )
        self.weak_loss_weight = float(weak_loss_weight)
        self.safe_kl_weight = float(safe_kl_weight)
        self.dist_kl_weight = float(dist_kl_weight)
        self.kl_temperature = float(kl_temperature)
        self.register_buffer("weak_classes", torch.tensor(tuple(weak_classes), dtype=torch.long), persistent=False)

        self.anchor_backbone = None
        self.anchor_seg_head = None
        if anchor_path is not None:
            self.anchor_backbone = build_model(backbone)
            self.anchor_seg_head = (
                nn.Linear(backbone_out_channels, num_classes)
                if num_classes > 0
                else nn.Identity()
            )
            self.anchor_load(
                torch.load(anchor_path, map_location="cpu", weights_only=False),
                keywords=anchor_keywords,
                replacements=anchor_replacements,
            )
            self.anchor_backbone.eval()
            self.anchor_seg_head.eval()
            for p in self.anchor_backbone.parameters():
                p.requires_grad = False
            for p in self.anchor_seg_head.parameters():
                p.requires_grad = False

    def anchor_load(self, checkpoint, keywords="", replacements=""):
        state_dict = checkpoint.get("state_dict", checkpoint)
        backbone_weight = OrderedDict()
        head_weight = OrderedDict()
        for key, value in state_dict.items():
            if not key.startswith("module."):
                key = "module." + key
            if keywords and keywords in key:
                key = key.replace(keywords, replacements, 1)
            key = key[7:]
            if key.startswith("backbone."):
                backbone_weight[key[9:]] = value
            elif key.startswith("seg_head."):
                head_weight[key[9:]] = value
        load_backbone_info = self.anchor_backbone.load_state_dict(backbone_weight, strict=False)
        load_head_info = self.anchor_seg_head.load_state_dict(head_weight, strict=False)
        print(f"Anchor backbone missing keys: {load_backbone_info[0]}")
        print(f"Anchor backbone unexpected keys: {load_backbone_info[1]}")
        print(f"Anchor head missing keys: {load_head_info[0]}")
        print(f"Anchor head unexpected keys: {load_head_info[1]}")

    def encode_logits(self, input_dict, backbone, seg_head, no_grad=False):
        def _forward():
            point = Point(input_dict)
            point = backbone(point)
            if isinstance(point, Point):
                while "pooling_parent" in point.keys():
                    assert "pooling_inverse" in point.keys()
                    parent = point.pop("pooling_parent")
                    inverse = point.pop("pooling_inverse")
                    parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                    point = parent
                feat = point.feat
            else:
                feat = point
            return seg_head(feat), point

        if no_grad:
            with torch.no_grad():
                return _forward()
        return _forward()

    def weak_mask(self, target):
        valid = target != -1
        mask = torch.zeros_like(valid, dtype=torch.bool)
        for cls in self.weak_classes.to(target.device):
            mask |= target == cls
        return valid & mask, valid & (~mask)

    def classsafe_loss(self, logits, anchor_logits, target):
        loss = logits.new_tensor(0.0)
        logs = {}
        weak, nonweak = self.weak_mask(target)
        if self.weak_loss_weight > 0 and weak.any():
            weak_ce = F.cross_entropy(logits[weak], target[weak], ignore_index=-1)
            loss = loss + self.weak_loss_weight * weak_ce
            logs["loss_weak_ce"] = weak_ce.detach()
        if anchor_logits is not None and self.safe_kl_weight > 0 and nonweak.any():
            temp = self.kl_temperature
            logp = F.log_softmax(logits[nonweak] / temp, dim=1)
            p0 = F.softmax(anchor_logits[nonweak] / temp, dim=1)
            safe_kl = F.kl_div(logp, p0, reduction="batchmean") * (temp * temp)
            loss = loss + self.safe_kl_weight * safe_kl
            logs["loss_safe_kl"] = safe_kl.detach()
        if anchor_logits is not None and self.dist_kl_weight > 0:
            valid = target != -1
            if valid.any():
                p = F.softmax(logits[valid], dim=1).mean(dim=0).clamp_min(1e-8)
                p0 = F.softmax(anchor_logits[valid], dim=1).mean(dim=0).clamp_min(1e-8)
                dist_kl = F.kl_div(p.log(), p0, reduction="sum")
                loss = loss + self.dist_kl_weight * dist_kl
                logs["loss_dist_kl"] = dist_kl.detach()
        return loss, logs

    def forward(self, input_dict, return_point=False):
        seg_logits, point = self.encode_logits(
            input_dict,
            self.backbone,
            self.seg_head,
            no_grad=self.freeze_backbone and not self.use_lora,
        )
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        anchor_logits = None
        if self.training and self.anchor_backbone is not None:
            anchor_logits, _ = self.encode_logits(
                input_dict,
                self.anchor_backbone,
                self.anchor_seg_head,
                no_grad=True,
            )

        if self.training:
            base_loss = self.criteria(seg_logits, input_dict["segment"])
            extra_loss, logs = self.classsafe_loss(seg_logits, anchor_logits, input_dict["segment"])
            return_dict["loss"] = base_loss + extra_loss
            return_dict["loss_base"] = base_loss.detach()
            return_dict.update(logs)
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
