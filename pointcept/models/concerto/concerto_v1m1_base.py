"""
Concerto V1M1

Author: Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from itertools import chain
from packaging import version
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter
from timm.layers import trunc_normal_
from transformers import AutoModel
from copy import deepcopy

import pointops
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils import (
    offset2batch,
    offset2bincount,
    batch2offset,
    bincount2offset,
)
from pointcept.utils.comm import get_world_size, all_gather
from pointcept.utils.scheduler import CosineScheduler


class OnlineCluster(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=4096,
        embed_channels=512,
        num_prototypes=4096,
        enable_mlp=True,
    ):
        super().__init__()
        if enable_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, embed_channels),
            )
        self.apply(self._init_weights)
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            self.prototype = torch.nn.utils.parametrizations.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.parametrizations.weight.original0.data.fill_(1)
            self.prototype.parametrizations.weight.original0.requires_grad = False

        else:
            self.prototype = torch.nn.utils.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.weight_g.data.fill_(1)
            self.prototype.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        if hasattr(self, "mlp"):
            feat = self.mlp(feat)
        eps = 1e-6 if feat.dtype == torch.float16 else 1e-12
        feat = nn.functional.normalize(feat, dim=-1, p=2, eps=eps)
        similarity = self.prototype(feat)
        return similarity


@MODELS.register_module("Concerto-v1m1")
class Concerto(PointModel):
    def __init__(
        self,
        image_weight_name,
        image_weight_path,
        backbone,
        head_in_channels,
        backbone_out_channels,
        embedding_channels,
        patch_w,
        patch_h,
        student_pretrained_path=None,
        teacher_pretrained_path=None,
        student_pretrained=False,
        head_hidden_channels=4096,
        head_embed_channels=512,
        head_num_prototypes=4096,
        enc2d_head_in_channels=384,
        enc2d_head_hidden_channels=4096,
        enc2d_head_embed_channels=256,
        enc2d_head_num_prototypes=384,
        teacher_custom=None,
        num_global_view=2,
        num_local_view=4,
        mask_size_start=0.1,
        mask_size_base=0.4,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
        mask_jitter=None,
        teacher_temp_start=0.04,
        teacher_temp_base=0.07,
        teacher_temp_warmup_ratio=0.05,
        student_temp=0.1,
        mask_loss_weight=2 / 10,
        roll_mask_loss_weight=2 / 10,
        unmask_loss_weight=4 / 10,
        enc2d_loss_weight=2 / 10,
        momentum_base=0.996,
        momentum_final=1,
        match_max_k=8,
        match_max_r=0.08,
        up_cast_level=2,
        enc2d_upcast_level=4,
        enc2d_cos_shift=True,
        sonata_model_type="offline",
        shortcut_probe=None,
    ):
        super(Concerto, self).__init__()
        assert sonata_model_type in ["online", "offline"]
        self.mask_loss_weight = mask_loss_weight
        self.roll_mask_loss_weight = roll_mask_loss_weight
        self.unmask_loss_weight = unmask_loss_weight
        self.enc2d_loss_weight = enc2d_loss_weight

        self.num_global_view = num_global_view
        self.num_local_view = num_local_view

        # masking and scheduler
        self.mask_size = mask_size_start
        self.mask_size_start = mask_size_start
        self.mask_size_base = mask_size_base
        self.mask_size_warmup_ratio = mask_size_warmup_ratio
        self.mask_size_scheduler = None

        self.mask_ratio = mask_ratio_start
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_base = mask_ratio_base
        self.mask_ratio_warmup_ratio = mask_ratio_warmup_ratio
        self.mask_ratio_scheduler = None

        self.mask_jitter = mask_jitter

        # temperature and scheduler
        self.teacher_temp = teacher_temp_start
        self.teacher_temp_start = teacher_temp_start
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_warmup_ratio = teacher_temp_warmup_ratio
        self.teacher_temp_scheduler = None
        self.student_temp = student_temp

        # momentum and scheduler
        self.momentum = momentum_base
        self.momentum_base = momentum_base
        self.momentum_final = momentum_final
        self.momentum_scheduler = None

        # dynamic matching
        self.match_max_k = match_max_k
        self.match_max_r = match_max_r

        # up cast level
        self.up_cast_level = up_cast_level
        self.enc2d_upcast_level = enc2d_upcast_level

        # one of unmask, mask, roll mask loss enable
        assert (
            unmask_loss_weight
            + mask_loss_weight
            + roll_mask_loss_weight
            + enc2d_loss_weight
            > 0
        )
        # roll mask loss need more than one global view
        assert num_global_view > 1 or roll_mask_loss_weight == 0
        # current roll mask only support two global views
        assert num_global_view == 1 or num_global_view == 2

        student_model_dict = dict()
        teacher_model_dict = dict()
        if teacher_custom is None:
            teacher_custom = {}
        student_backbone = build_model(backbone)
        if student_pretrained_path != None:
            print("Load pretrained student model")
            student_backbone = self.load_sonata(
                student_backbone, path=student_pretrained_path
            )

        # turn off parameters like drop path for teacher model
        backbone.update(teacher_custom)

        teacher_backbone = build_model(backbone)
        if sonata_model_type == "offline":
            teacher_backbone = self.load_sonata(
                teacher_backbone, path=teacher_pretrained_path
            )
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        if self.enc2d_loss_weight > 0:
            self.patch_h = patch_h
            self.patch_w = patch_w
            self.image_weight_name = image_weight_name

            # Load Model
            self.enc2d_model = self.load_enc2d(image_weight_name, image_weight_path)
            self.enc2d_model.requires_grad_(False)
            self._num_channels = enc2d_head_in_channels
            self.patch_proj = torch.nn.Linear(backbone_out_channels, self._num_channels)

            enc2d_head = partial(
                OnlineCluster,
                in_channels=enc2d_head_in_channels,
                hidden_channels=enc2d_head_hidden_channels,
                embed_channels=enc2d_head_in_channels,
                num_prototypes=enc2d_head_num_prototypes,
            )
            enc2d_head_ = partial(
                OnlineCluster,
                in_channels=enc2d_head_in_channels,
                hidden_channels=enc2d_head_hidden_channels,
                embed_channels=enc2d_head_in_channels,
                num_prototypes=enc2d_head_num_prototypes,
                enable_mlp=False,
            )

            self.enc2d_head_student = enc2d_head()
            self.enc2d_head_teacher = enc2d_head_()
            self.enc2d_head_student.prototype.load_state_dict(
                self.enc2d_head_teacher.prototype.state_dict()
            )
            for p in self.enc2d_head_teacher.parameters():
                p.requires_grad = False

        head = partial(
            OnlineCluster,
            in_channels=head_in_channels,
            hidden_channels=head_hidden_channels,
            embed_channels=head_embed_channels,
            num_prototypes=head_num_prototypes,
        )
        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            student_model_dict["mask_head"] = head()
            teacher_model_dict["mask_head"] = head()
        if self.unmask_loss_weight > 0:
            student_model_dict["unmask_head"] = head()
            teacher_model_dict["unmask_head"] = head()
        if (
            self.enc2d_loss_weight > 0
            and self.unmask_loss_weight
            + self.mask_loss_weight
            + self.roll_mask_loss_weight
            == 0
        ):
            student_model_dict["unmask_head"] = head()
            teacher_model_dict["unmask_head"] = head()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        for k, v in self.student.items():
            if "head" in k:
                self.teacher[k].load_state_dict(self.student[k].state_dict())
        if sonata_model_type == "online":
            self.teacher.backbone.load_state_dict(self.student.backbone.state_dict())
        for n, p in self.teacher.named_parameters():
            p.requires_grad = False

        self.enc2d_cos_shift = enc2d_cos_shift
        self.sonata_model_type = sonata_model_type
        self.shortcut_probe = {
            "mode": "none",
            "freeze_student_backbone": False,
            "zero_color": False,
            "zero_normal": False,
            "coord_jitter_std": 0.0,
            "coord_jitter_clip": None,
            "coord_normalize": True,
            "coord_probe_hidden_channels": 512,
            "coord_prior_loss_weight": 1.0,
            "shuffle_correspondence": False,
        }
        if shortcut_probe is not None:
            self.shortcut_probe.update(shortcut_probe)

        if self.enc2d_loss_weight > 0 and self.shortcut_probe["mode"] == "coord_mlp":
            self.coord_probe = nn.Sequential(
                nn.Linear(3, self.shortcut_probe["coord_probe_hidden_channels"]),
                nn.GELU(),
                nn.Linear(
                    self.shortcut_probe["coord_probe_hidden_channels"],
                    backbone_out_channels,
                ),
            )
            self.coord_probe.apply(OnlineCluster._init_weights)

        if (
            self.enc2d_loss_weight > 0
            and self.shortcut_probe["mode"] == "coord_residual_target"
        ):
            self.coord_prior = nn.Sequential(
                nn.Linear(3, self.shortcut_probe["coord_probe_hidden_channels"]),
                nn.GELU(),
                nn.Linear(
                    self.shortcut_probe["coord_probe_hidden_channels"],
                    self._num_channels,
                ),
            )
            self.coord_prior.apply(OnlineCluster._init_weights)

        if self.shortcut_probe["freeze_student_backbone"]:
            self.student.backbone.requires_grad_(False)
        self._shortcut_probe_logs = set()

    def _apply_shortcut_probe_to_inputs(self, data_dict):
        if self.shortcut_probe is None:
            return data_dict
        if not any(
            [
                self.shortcut_probe["zero_color"],
                self.shortcut_probe["zero_normal"],
                self.shortcut_probe["coord_jitter_std"] > 0,
            ]
        ):
            return data_dict

        data_dict = dict(data_dict)
        clone_keys = [
            "global_feat",
            "local_feat",
            "global_coord",
            "local_coord",
            "global_origin_coord",
            "local_origin_coord",
        ]
        for key in clone_keys:
            if key in data_dict and torch.is_tensor(data_dict[key]):
                data_dict[key] = data_dict[key].clone()

        for feat_key in ("global_feat", "local_feat"):
            if feat_key in data_dict and torch.is_tensor(data_dict[feat_key]):
                if self.shortcut_probe["zero_color"] and data_dict[feat_key].shape[1] >= 6:
                    data_dict[feat_key][:, 3:6] = 0
                if self.shortcut_probe["zero_normal"] and data_dict[feat_key].shape[1] >= 9:
                    data_dict[feat_key][:, 6:9] = 0

        coord_jitter_std = self.shortcut_probe["coord_jitter_std"]
        if coord_jitter_std > 0:
            coord_jitter_clip = self.shortcut_probe["coord_jitter_clip"]
            coord_pairs = (
                ("global_coord", "global_origin_coord", "global_feat"),
                ("local_coord", "local_origin_coord", "local_feat"),
            )
            for coord_key, origin_coord_key, feat_key in coord_pairs:
                if coord_key not in data_dict or not torch.is_tensor(data_dict[coord_key]):
                    continue
                noise = torch.randn_like(data_dict[coord_key]).mul(coord_jitter_std)
                if coord_jitter_clip is not None:
                    noise = noise.clamp(-coord_jitter_clip, coord_jitter_clip)
                data_dict[coord_key] = data_dict[coord_key] + noise
                if origin_coord_key in data_dict and torch.is_tensor(data_dict[origin_coord_key]):
                    data_dict[origin_coord_key] = data_dict[origin_coord_key] + noise
                if feat_key in data_dict and torch.is_tensor(data_dict[feat_key]) and data_dict[feat_key].shape[1] >= 3:
                    data_dict[feat_key][:, :3] = data_dict[coord_key]

        return data_dict

    @staticmethod
    def _scene_normalize_coord(coord, offset):
        splits = coord.split(offset2bincount(offset).tolist())
        coord_list = []
        for split in splits:
            mean = split.mean(dim=0, keepdim=True)
            std = split.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            coord_list.append((split - mean) / std)
        return torch.cat(coord_list, dim=0)

    @staticmethod
    def _shuffle_correspondence_by_offset(correspondence, offset):
        shuffled = correspondence.clone()
        starts = torch.cat(
            [torch.zeros(1, device=offset.device, dtype=offset.dtype), offset[:-1]], dim=0
        )
        for start, end in zip(starts.tolist(), offset.tolist()):
            if end - start <= 1:
                continue
            perm = torch.randperm(end - start, device=correspondence.device) + start
            shuffled[start:end] = shuffled[perm]
        return shuffled

    @staticmethod
    def _sample_derangement(num_items, device):
        if num_items <= 1:
            raise ValueError(
                "cross_scene_target_swap requires at least two scenes with valid teacher targets per batch."
            )
        identity = torch.arange(num_items, device=device)
        for _ in range(32):
            perm = torch.randperm(num_items, device=device)
            if not torch.any(perm == identity):
                return perm
        return torch.roll(identity, shifts=1, dims=0)

    @staticmethod
    def _resample_rows_with_replacement(rows, target_length):
        if rows.shape[0] == 0:
            raise ValueError("Cannot resample teacher targets from an empty source block.")
        if rows.shape[0] == target_length:
            return rows
        sample_index = torch.randint(
            rows.shape[0], (target_length,), device=rows.device
        )
        return rows[sample_index]

    def _log_shortcut_probe_once(self, tag, message):
        if tag not in self._shortcut_probe_logs:
            self._shortcut_probe_logs.add(tag)
            print(message)

    def _apply_global_target_permutation(
        self, teacher_target, mode_name="global_target_permutation"
    ):
        if teacher_target.shape[0] <= 1:
            self._log_shortcut_probe_once(
                f"{mode_name}:skip",
                f"[shortcut_probe] {mode_name} skip=identity rows={teacher_target.shape[0]}",
            )
            return teacher_target
        perm = self._sample_derangement(teacher_target.shape[0], teacher_target.device)
        self._log_shortcut_probe_once(
            mode_name,
            f"[shortcut_probe] {mode_name} self_assignments=0 rows={teacher_target.shape[0]}",
        )
        return teacher_target[perm]

    def _apply_group_target_swap(self, teacher_target, target_group_index, mode_name):
        unique_group_index = torch.unique(target_group_index, sorted=True)
        if unique_group_index.numel() <= 1:
            self._log_shortcut_probe_once(
                f"{mode_name}:fallback",
                f"[shortcut_probe] {mode_name} fallback=global_target_permutation valid_groups=1",
            )
            return self._apply_global_target_permutation(
                teacher_target, mode_name=f"{mode_name}:fallback_permutation"
            )

        perm = self._sample_derangement(unique_group_index.numel(), teacher_target.device)
        swapped = teacher_target.clone()
        swap_log_parts = []
        for dest_pos, src_pos in enumerate(perm.tolist()):
            dest_group = unique_group_index[dest_pos]
            src_group = unique_group_index[src_pos]
            dest_mask = target_group_index == dest_group
            source_rows = teacher_target[target_group_index == src_group]
            dest_count = int(dest_mask.sum().item())
            swapped[dest_mask] = self._resample_rows_with_replacement(
                source_rows, dest_count
            )
            swap_log_parts.append(
                f"{int(dest_group.item())}<-{int(src_group.item())}:{dest_count}/{int(source_rows.shape[0])}"
            )

        self._log_shortcut_probe_once(
            mode_name,
            f"[shortcut_probe] {mode_name} self_assignments=0 mapping={', '.join(swap_log_parts)}",
        )
        return swapped

    def _apply_cross_scene_target_swap(self, teacher_target, target_batch_index):
        return self._apply_group_target_swap(
            teacher_target,
            target_batch_index,
            mode_name="cross_scene_target_swap",
        )

    def _apply_cross_image_target_swap(self, teacher_target, target_image_index):
        return self._apply_group_target_swap(
            teacher_target,
            target_image_index,
            mode_name="cross_image_target_swap",
        )

    def _apply_prepool_global_feature_index_permutation(self, feature_index):
        return self._apply_global_target_permutation(
            feature_index, mode_name="prepool_global_feature_index_permutation"
        )

    def load_enc2d(self, model_name, model_weight):
        model = AutoModel.from_pretrained(model_weight, trust_remote_code=True)
        return model.eval()

    def load_sonata(self, model, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda())
        weight = {}
        whether_weight = False
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
            for key, value in checkpoint.items():
                if "module.student.backbone." in key:
                    whether_weight = True
                    key = key.replace("module.student.backbone.", "module.")
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                    weight[key] = value
        if whether_weight:
            load_state_info = model.load_state_dict(weight)
        else:
            load_state_info = model.load_state_dict(checkpoint)
        print(f"Missing keys: {load_state_info[0]}")
        print(f"Unexpected keys: {load_state_info[1]}")
        return model

    @torch.no_grad()
    def ENC2D_forward(self, x):
        # RADIO
        if "radio" in self.image_weight_name:
            summary, features = self.enc2d_model(x)
            features = features.reshape(
                -1, self.patch_h * self.patch_w, self._num_channels
            )
            return features
        # SigLIPv2
        if hasattr(self.enc2d_model, "vision_model"):
            outputs = self.enc2d_model.vision_model(x)
            features = outputs.last_hidden_state
        # DINOv2.5
        else:
            outputs = self.enc2d_model(x)
            features = outputs.last_hidden_state[:, -self.patch_h * self.patch_w :, :]
        return features

    def before_train(self):
        # make ModelHook after CheckPointLoader
        total_steps = self.trainer.cfg.scheduler.total_steps
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        # mask size scheduler
        self.mask_size_scheduler = CosineScheduler(
            start_value=self.mask_size_start,
            base_value=self.mask_size_base,
            final_value=self.mask_size_base,
            warmup_iters=int(total_steps * self.mask_size_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_size_scheduler.iter = curr_step

        # mask ratio scheduler
        self.mask_ratio_scheduler = CosineScheduler(
            start_value=self.mask_ratio_start,
            base_value=self.mask_ratio_base,
            final_value=self.mask_ratio_base,
            warmup_iters=int(total_steps * self.mask_ratio_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_ratio_scheduler.iter = curr_step

        # teacher temperature scheduler
        self.teacher_temp_scheduler = CosineScheduler(
            start_value=self.teacher_temp_start,
            base_value=self.teacher_temp_base,
            final_value=self.teacher_temp_base,
            warmup_iters=int(total_steps * self.teacher_temp_warmup_ratio),
            total_iters=total_steps,
        )
        self.teacher_temp_scheduler.iter = curr_step

        # momentum scheduler
        self.momentum_scheduler = CosineScheduler(
            base_value=self.momentum_base,
            final_value=self.momentum_final,
            total_iters=total_steps,
        )
        self.momentum_scheduler.iter = curr_step

    def before_step(self):
        # update parameters from schedulers
        self.mask_size = self.mask_size_scheduler.step()
        self.mask_ratio = self.mask_ratio_scheduler.step()
        self.teacher_temp = self.teacher_temp_scheduler.step()
        self.momentum = self.momentum_scheduler.step()

        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                "params/mask_size",
                self.mask_size,
                self.mask_size_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/mask_ratio",
                self.mask_ratio,
                self.mask_ratio_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/teacher_temp",
                self.teacher_temp,
                self.teacher_temp_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/momentum",
                self.momentum,
                self.momentum_scheduler.iter,
            )

    def after_step(self):
        # pass
        # EMA update teacher
        with torch.no_grad():
            m = self.momentum
            if self.sonata_model_type == "online":
                student_param_list = list(self.student.backbone.parameters())
                teacher_param_list = list(self.teacher.backbone.parameters())
                torch._foreach_mul_(teacher_param_list, m)
                torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

            student_param_list = [
                p for n, p in self.student.named_parameters() if "head" in n
            ]
            teacher_param_list = [
                p for n, p in self.teacher.named_parameters() if "head" in n
            ]
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

            if self.enc2d_loss_weight > 0:
                enc2d_student_param_list = [
                    p
                    for n, p in self.enc2d_head_student.named_parameters()
                    if "prototype" in n
                ]
                enc2d_teacher_param_list = [
                    p
                    for n, p in self.enc2d_head_teacher.named_parameters()
                    if "prototype" in n
                ]
                torch._foreach_copy_(enc2d_teacher_param_list, enc2d_student_param_list)

    @staticmethod
    def sinkhorn_knopp(feat, temp, num_iter=3):
        feat = feat.float()
        q = torch.exp(feat / temp).t()
        n = sum(all_gather(q.shape[1]))  # number of samples to assign
        k = q.shape[0]  # number of prototypes

        # make the matrix sums to 1
        sum_q = q.sum()
        if get_world_size() > 1:
            dist.all_reduce(sum_q)
        q = q / sum_q

        for i in range(num_iter):
            # normalize each row: total weight per prototype must be 1/k
            q_row_sum = q.sum(dim=1, keepdim=True)
            if get_world_size() > 1:
                dist.all_reduce(q_row_sum)
            q = q / q_row_sum / k

            # normalize each column: total weight per sample must be 1/n
            q = q / q.sum(dim=0, keepdim=True) / n

        q *= n  # the columns must sum to 1 so that Q is an assignment
        return q.t()

    def generate_mask(self, coord, offset):
        batch = offset2batch(offset)
        mask_size = self.mask_size
        mask_ratio = self.mask_ratio

        # Grouping points with grid patch
        min_coord = torch_scatter.segment_coo(coord, batch, reduce="min")
        grid_coord = ((coord - min_coord[batch]) // mask_size).int()
        grid_coord = torch.cat([batch.unsqueeze(-1), grid_coord], dim=-1)
        unique, point_cluster, counts = torch.unique(
            grid_coord, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        patch_num = unique.shape[0]
        mask_patch_num = int(patch_num * mask_ratio)
        patch_index = torch.randperm(patch_num, device=coord.device)
        mask_patch_index = patch_index[:mask_patch_num]
        point_mask = torch.isin(point_cluster, mask_patch_index)
        return point_mask, point_cluster

    @torch.no_grad()
    def match_neighbour(
        self,
        view1_coord,
        view1_offset,
        view2_coord,
        view2_offset,
    ):
        index2, distance = pointops.knn_query(
            1,
            view2_coord.float(),
            view2_offset.int(),
            view1_coord.float(),
            view1_offset.int(),
        )
        index1 = torch.arange(
            index2.shape[0], device=index2.device, dtype=torch.long
        ).unsqueeze(-1)
        index = torch.cat([index1, index2], dim=-1)[
            distance.squeeze(-1) < self.match_max_r
        ]
        return index

    @torch.no_grad()
    def roll_point(self, point):
        n = self.num_global_view
        # [pc1, pc1', pc2, pc2'] -> [pc1', pc1, pc2', pc2], only support num_global_view == 2
        bs = len(point.offset) // self.num_global_view
        data_dict = {}
        for key in point.keys():
            if key in ["feat", "coord", "origin_coord", "batch"]:
                value = point[key].split(offset2bincount(point.offset).tolist())
                value = chain(*[value[n * b : n * (b + 1)][::-1] for b in range(bs)])
                if key == "batch":
                    value = [torch.ones_like(v) * i for i, v in enumerate(value)]
                data_dict[key] = torch.cat(list(value), dim=0)
        return Point(data_dict)

    def up_cast(self, point, upcast_level=None):
        if upcast_level is None:
            upcast_level = self.up_cast_level
        else:
            upcast_level = upcast_level
        for _ in range(upcast_level):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    @staticmethod
    def pool_corr(point, correspondence):
        inverse_list = []
        idx_ptr_list = []
        point_feat = dict(offset=point.offset, feat=point.feat, coord=point.coord)
        while "pooling_parent" in point.keys():
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            assert "idx_ptr" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            idx_ptr = point.pop("idx_ptr")
            inverse_list.append(inverse)
            idx_ptr_list.append(idx_ptr)
            point = parent
        inverse_list.reverse()
        idx_ptr_list.reverse()
        for inverse, idx_ptr in zip(inverse_list, idx_ptr_list):
            _, indices = torch.sort(inverse)
            img_num = correspondence.shape[1]
            correspondence_all = []
            if img_num == 0:
                correspondence_all = -torch.ones((idx_ptr.shape[0] - 1, 0, 2)).cuda()
            else:
                for img_id in range(img_num):
                    mask = torch.all(
                        correspondence[:, img_id] != torch.tensor([-1, -1]).cuda(),
                        dim=1,
                    ).float()
                    counts = torch_scatter.segment_csr(
                        mask[indices], idx_ptr, reduce="sum"
                    )
                    counts[counts == 0] = 100000
                    correspondence_img = deepcopy(correspondence[:, img_id])
                    correspondence_img[correspondence_img == -1] = 0
                    mask_sum = torch_scatter.segment_csr(
                        correspondence_img[indices], idx_ptr, reduce="sum"
                    )
                    mask_sum = mask_sum / counts.unsqueeze(1)
                    mask_sum[counts == 100000] = -1
                    correspondence_all.append(mask_sum)
                correspondence_all = torch.stack(correspondence_all, dim=1)
            correspondence = correspondence_all
        point_feat["correspondence"] = correspondence
        point_feat = Point(point_feat)
        return point_feat

    def forward(self, data_dict, return_point=False):
        data_dict = self._apply_shortcut_probe_to_inputs(data_dict)
        enc2d_only = (
            self.enc2d_loss_weight > 0
            and self.mask_loss_weight + self.roll_mask_loss_weight + self.unmask_loss_weight
            == 0
        )
        if return_point:
            point = self.teacher.backbone(data_dict)
            for _ in range(self.up_cast_level):
                assert "pooling_parent" in point.keys()
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            return dict(point=point)

        # prepare global_point, mask_global_point, local_point
        with torch.no_grad():
            # global_point & masking
            global_point = Point(
                feat=data_dict["global_feat"],
                coord=data_dict["global_coord"],
                origin_coord=data_dict["global_origin_coord"],
                offset=data_dict["global_offset"],
                grid_size=data_dict["grid_size"][0],
            )

            global_mask, global_cluster = self.generate_mask(
                global_point.coord, global_point.offset
            )
            mask_global_coord = global_point.coord.clone().detach()
            if self.mask_jitter is not None:
                mask_global_coord[global_mask] += torch.clip(
                    torch.randn_like(mask_global_coord[global_mask]).mul(
                        self.mask_jitter
                    ),
                    max=self.mask_jitter * 2,
                )

            mask_global_point = Point(
                feat=data_dict["global_feat"],
                coord=mask_global_coord,
                origin_coord=data_dict["global_origin_coord"],
                mask=global_mask,
                offset=data_dict["global_offset"],
                grid_size=data_dict["grid_size"][0],
            )
            major_view_correspondence = data_dict["global_correspondence"]

            # create result dictionary for return
            result_dict = dict(loss=[])
            if not enc2d_only:
                # local point & matching
                local_point = Point(
                    feat=data_dict["local_feat"],
                    coord=data_dict["local_coord"],
                    origin_coord=data_dict["local_origin_coord"],
                    offset=data_dict["local_offset"],
                    grid_size=data_dict["grid_size"][0],
                )

                # teacher forward
                global_point_ = self.teacher.backbone(global_point)
                global_point_ = self.up_cast(global_point_)
                # only use one shared head for both mask and unmask
                # priority: mask (global) > unmask (local)
                if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
                    global_point_.feat = self.teacher.mask_head(global_point_.feat)
                else:
                    global_point_.feat = self.teacher.unmask_head(global_point_.feat)

        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            # student forward
            mask_global_point_ = self.student.backbone(mask_global_point)
            mask_global_point_ = self.up_cast(mask_global_point_)
            mask_pred_sim = self.student.mask_head(mask_global_point_.feat)

            if self.mask_loss_weight > 0:
                with torch.no_grad():
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        global_point_.origin_coord,
                        global_point_.offset,
                    )
                    # teacher forward
                    mask_target_sim = self.sinkhorn_knopp(
                        global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                # loss
                mask_loss = -torch.sum(
                    mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )

                mask_loss = torch_scatter.segment_coo(
                    mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["mask_loss"] = mask_loss
                result_dict["loss"].append(mask_loss * self.mask_loss_weight)

            if self.roll_mask_loss_weight > 0:
                roll_global_point_ = self.roll_point(global_point_)
                with torch.no_grad():
                    # match index for pred and roll target
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        roll_global_point_.origin_coord,
                        roll_global_point_.offset,
                    )
                    # teacher forward
                    roll_mask_target_sim = self.sinkhorn_knopp(
                        roll_global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                roll_mask_loss = -torch.sum(
                    roll_mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                roll_mask_loss = torch_scatter.segment_coo(
                    roll_mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["roll_mask_loss"] = roll_mask_loss
                result_dict["loss"].append(roll_mask_loss * self.roll_mask_loss_weight)
        if self.unmask_loss_weight > 0:
            # student forward
            local_point_ = self.student.backbone(local_point)
            local_point_ = self.up_cast(local_point_)
            unmask_pred_sim = self.student.unmask_head(local_point_.feat)
            with torch.no_grad():
                principal_view_mask = global_point_.batch % self.num_global_view == 0
                principal_view_batch = (
                    global_point_.batch[principal_view_mask] // self.num_global_view
                )
                match_index = self.match_neighbour(
                    local_point_.origin_coord,
                    local_point_.offset[self.num_local_view - 1 :: self.num_local_view],
                    global_point_.origin_coord[principal_view_mask],
                    batch2offset(principal_view_batch),
                )
                # teacher forward
                unmask_target_sim = self.sinkhorn_knopp(
                    global_point_.feat[principal_view_mask][match_index[:, 1]],
                    self.teacher_temp,
                )
            # loss
            unmask_loss = -torch.sum(
                unmask_target_sim
                * F.log_softmax(
                    unmask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                ),
                dim=-1,
            )
            unmask_loss = torch_scatter.segment_coo(
                unmask_loss,
                index=local_point_.batch[match_index[:, 0]],
                reduce="mean",
            ).mean()
            result_dict["unmask_loss"] = unmask_loss
            result_dict["loss"].append(unmask_loss * self.unmask_loss_weight)
        if self.enc2d_loss_weight > 0:
            if enc2d_only:
                mask_global_point_ = self.student.backbone(mask_global_point)
                mask_global_point_ = self.up_cast(mask_global_point_)
            elif self.mask_loss_weight == 0 or self.roll_mask_loss_weight == 0:
                mask_global_point_ = self.student.backbone(mask_global_point)
                mask_global_point_ = self.up_cast(mask_global_point_)
            mask_global_point_enc2d = self.up_cast(
                mask_global_point_,
                upcast_level=self.enc2d_upcast_level - self.up_cast_level,
            )
            to_feature = self.pool_corr(
                mask_global_point_enc2d, major_view_correspondence
            )
            data_dict_global_offset = torch.cat(
                [torch.tensor([0]).cuda(), to_feature["offset"]], dim=0
            )
            enc2d_count = (
                data_dict_global_offset[
                    1 : len(data_dict_global_offset) : self.num_global_view
                ]
                - data_dict_global_offset[
                    0 : len(data_dict_global_offset) - 1 : self.num_global_view
                ]
            )
            enc2d_offset = torch.cat(
                [torch.tensor([0]).cuda(), torch.cumsum(enc2d_count, dim=0)]
            )
            enc2d_mask = torch.cat(
                [
                    torch.arange(0, c, device=enc2d_count.device)
                    + data_dict_global_offset[i * self.num_global_view]
                    for i, c in enumerate(enc2d_count)
                ],
                dim=0,
            )

            offset_points_3d = enc2d_offset[1:]
            batch_points_3d = offset2batch(offset_points_3d)
            imgs = data_dict["images"]
            pooled_coord = to_feature["coord"][enc2d_mask]
            if self.shortcut_probe["coord_normalize"]:
                pooled_coord = self._scene_normalize_coord(pooled_coord, enc2d_offset[1:])
            if self.shortcut_probe["mode"] == "coord_mlp":
                feature3d = self.coord_probe(pooled_coord)
            else:
                feature3d = to_feature["feat"][enc2d_mask]
            enc2d_global_mask = global_mask[enc2d_mask]
            correspondence = to_feature["correspondence"][enc2d_mask]
            if self.shortcut_probe["shuffle_correspondence"]:
                correspondence = self._shuffle_correspondence_by_offset(correspondence, enc2d_offset[1:])
            v0 = correspondence.shape[1]
            mask = torch.any(correspondence != torch.tensor([-1, -1]).cuda(), dim=2)
            enc2d_global_mask = enc2d_global_mask.unsqueeze(1).expand(-1, v0)
            valid_index = torch.where(mask)  # 0: 3d points index, 1: view index

            bincount_img_num = data_dict["img_num"]
            offset_img_num = bincount2offset(bincount_img_num)
            total_img_num = offset_img_num[-1]

            if total_img_num > 0:
                # expand
                with torch.no_grad():
                    feature2d = self.ENC2D_forward(imgs)
                    feature2d = feature2d.contiguous().view(-1, feature2d.shape[-1])
                    feature2d_mask = feature2d

                offset_img_num = torch.cat([torch.tensor([0]).cuda(), offset_img_num])[
                    :-1
                ]
                batch_index = batch_points_3d[valid_index[0]]
                batch_img_num = offset_img_num[batch_index]

                feature3d_mask = feature3d[valid_index[0]]
                feature_coord_mask = pooled_coord[valid_index[0]]

                feature_index = torch.cat(
                    [
                        batch_img_num.unsqueeze(-1),
                        valid_index[1].unsqueeze(-1),
                        correspondence[valid_index],
                    ],
                    dim=-1,
                )
                feature_index = feature_index.long()
                feature_index = (
                    feature_index[:, 0] * self.patch_h * self.patch_w
                    + feature_index[:, 1] * self.patch_h * self.patch_w
                    + feature_index[:, 2] * self.patch_w
                    + feature_index[:, 3]
                )
                if self.shortcut_probe["mode"] == "prepool_global_feature_index_permutation":
                    feature_index = self._apply_prepool_global_feature_index_permutation(
                        feature_index
                    )

                feature_index, inverse_index = torch.unique(
                    feature_index.long(), sorted=True, return_inverse=True
                )
                feature3d_mask = torch_scatter.scatter_mean(
                    feature3d_mask,
                    inverse_index,
                    dim=0,
                    dim_size=feature_index.shape[0],
                )
                feature_coord_mask = torch_scatter.scatter_mean(
                    feature_coord_mask,
                    inverse_index,
                    dim=0,
                    dim_size=feature_index.shape[0],
                )
                feature3d_mask = self.patch_proj(feature3d_mask)
                feature_batch_index = torch_scatter.scatter_min(
                    batch_index.long(),
                    inverse_index,
                    dim=0,
                    dim_size=feature_index.shape[0],
                )[0]
                feature_image_index = torch.div(
                    feature_index,
                    self.patch_h * self.patch_w,
                    rounding_mode="floor",
                )
                feature2d_mask = feature2d_mask[feature_index]
                if self.shortcut_probe["mode"] == "cross_scene_target_swap":
                    feature2d_mask = self._apply_cross_scene_target_swap(
                        feature2d_mask, feature_batch_index
                    )
                elif self.shortcut_probe["mode"] == "cross_image_target_swap":
                    feature2d_mask = self._apply_cross_image_target_swap(
                        feature2d_mask, feature_image_index
                    )
                elif self.shortcut_probe["mode"] == "global_target_permutation":
                    feature2d_mask = self._apply_global_target_permutation(
                        feature2d_mask
                    )

                raw_feature2d_mask = feature2d_mask
                coord_prior_pred = None
                if self.shortcut_probe["mode"] == "coord_residual_target":
                    coord_prior_pred = self.coord_prior(feature_coord_mask)
                    feature2d_mask = raw_feature2d_mask - coord_prior_pred.detach()

                if self.enc2d_cos_shift:
                    feature2d_mask = feature2d_mask - feature2d_mask.mean(
                        dim=-1, keepdim=True
                    )
                    feature3d_mask = feature3d_mask - feature3d_mask.mean(
                        dim=-1, keepdim=True
                    )
                    if coord_prior_pred is not None:
                        raw_feature2d_mask = raw_feature2d_mask - raw_feature2d_mask.mean(
                            dim=-1, keepdim=True
                        )
                        coord_prior_pred = coord_prior_pred - coord_prior_pred.mean(
                            dim=-1, keepdim=True
                        )
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                loss = (1 - cos(feature2d_mask, feature3d_mask)).mean() * 10

                result_dict["enc2d_loss"] = loss
                result_dict["loss"].append(loss * self.enc2d_loss_weight)
                if coord_prior_pred is not None:
                    coord_prior_loss = (1 - cos(raw_feature2d_mask, coord_prior_pred)).mean() * 10
                    result_dict["coord_prior_loss"] = coord_prior_loss
                    result_dict["loss"].append(
                        coord_prior_loss * self.shortcut_probe["coord_prior_loss_weight"]
                    )
                del (
                    feature2d,
                    feature3d,
                    feature2d_mask,
                    feature3d_mask,
                    feature_coord_mask,
                    feature_batch_index,
                    feature_image_index,
                    correspondence,
                    feature_index,
                )
            elif (
                self.mask_loss_weight
                + self.unmask_loss_weight
                + self.roll_mask_loss_weight
                > 0
            ):
                result_ssl_loss = sum(result_dict["loss"]) / (
                    self.mask_loss_weight
                    + self.unmask_loss_weight
                    + self.roll_mask_loss_weight
                )
                result_dict["enc2d_loss"] = result_ssl_loss
                result_dict["loss"].append(result_ssl_loss * self.enc2d_loss_weight)
            else:
                zero_loss = feature3d.sum() * 0.0
                result_dict["enc2d_loss"] = zero_loss
                result_dict["loss"].append(zero_loss)
        result_dict["loss"] = sum(result_dict["loss"])

        if get_world_size() > 1:
            for loss_id, loss in result_dict.items():
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        return result_dict
