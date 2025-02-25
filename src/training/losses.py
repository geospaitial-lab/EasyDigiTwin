import json

import cv2
import math
import os.path

import numpy as np
import torch
import torchvision.transforms

from src.utils.callback_utils import Callback
from src.utils.general_utils import relative_image_gradient_magnitude, minpool_grayscale, \
    squared_normalized_cross_correlation
from src.utils.interfaces import Render, TargetRender, RenderInfo
from torchmetrics.functional.regression import pearson_corrcoef
from torchmetrics.functional.image import (structural_similarity_index_measure,
                                           multiscale_structural_similarity_index_measure)
from torchvision.transforms.functional import resize

from src.utils.ray_utils import info_to_rays, reproject_points_from_view, project_points, get_intrinsics_matrix


@torch.no_grad()
def sample_depth_loss_indices(depths, width, height, valid_mask, n_samples, rank_patch_size=100, knn_crop_radius=3):
    samples_y = torch.randint(0, int(height - rank_patch_size), (n_samples, 1), device=depths.device)
    samples_x = torch.randint(0, int(width - rank_patch_size), (n_samples, 1), device=depths.device)

    samples_y = samples_y + torch.randint(0, int(rank_patch_size), (n_samples, 2), device=depths.device)
    samples_x = samples_x + torch.randint(0, int(rank_patch_size), (n_samples, 2), device=depths.device)

    sample_idx = samples_y * int(width) + samples_x
    sampled_depth = depths.squeeze(0)[sample_idx]

    top_n_to_consider = int(0.3 * (knn_crop_radius * 2 + 1) ** 2 + 1)

    padded_depth = torch.nn.functional.pad(depths.reshape(int(height), int(width)), (knn_crop_radius,
                                                                                     knn_crop_radius,
                                                                                     knn_crop_radius,
                                                                                     knn_crop_radius), value=-1e6)

    crop_idx_y = samples_y[..., None] - knn_crop_radius + torch.arange(knn_crop_radius * 2 + 1,
                                                                       device=depths.device
                                                                       ).repeat_interleave(knn_crop_radius * 2 + 1
                                                                                           ).repeat(n_samples, 2, 1)
    padded_crop_idx_y = crop_idx_y + knn_crop_radius
    crop_idx_x = samples_x[..., None] - knn_crop_radius + torch.arange(knn_crop_radius * 2 + 1,
                                                                       device=depths.device
                                                                       ).repeat(n_samples, 2, knn_crop_radius * 2 + 1)
    padded_crop_idx_x = crop_idx_x + knn_crop_radius
    depth_crops = padded_depth[padded_crop_idx_y, padded_crop_idx_x]

    sorted_crop_idx = torch.argsort(torch.abs(depth_crops - sampled_depth[..., None]), dim=-1)
    neighbour_sample_idx = torch.randint(1, top_n_to_consider, (n_samples, 2, 1), device=depths.device)

    neigbours_idx_relative = torch.gather(sorted_crop_idx, -1, neighbour_sample_idx).squeeze(-1)

    neighbours_y = (samples_y - knn_crop_radius) + neigbours_idx_relative // (knn_crop_radius * 2 + 1)
    neighbours_x = (samples_x - knn_crop_radius) + neigbours_idx_relative % (knn_crop_radius * 2 + 1)

    neighbours_idx = neighbours_y * int(width) + neighbours_x

    if valid_mask is not None:
        samples_mask = valid_mask.squeeze(0)[sample_idx].any(dim=-1).all(dim=-1)
        neighbours_mask = valid_mask.squeeze(0)[neighbours_idx].any(dim=-1).all(dim=-1)

        full_mask = samples_mask * neighbours_mask

        filtered_sample_idx = sample_idx[full_mask]
        filtered_neighbours_idx = neighbours_idx[full_mask]
        filtered_sampled_depth = sampled_depth[full_mask]
    else:
        filtered_sample_idx = sample_idx
        filtered_neighbours_idx = neighbours_idx
        filtered_sampled_depth = sampled_depth

    order_idx = torch.argsort(filtered_sampled_depth, dim=-1, descending=True)
    sorted_sample_idx = torch.gather(filtered_sample_idx, -1, order_idx)
    sorted_neighbours_idx = torch.gather(filtered_neighbours_idx, -1, order_idx)

    idx = torch.cat([sorted_sample_idx, sorted_neighbours_idx], dim=-1).reshape(-1)

    return idx


class _RangeLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, range_min=0, range_max=1):
        zero = torch.zeros(1, device=values.device)
        loss = torch.heaviside(-(values - range_min), zero) * (values - range_min) ** 2 \
               + torch.heaviside(values - range_max, zero) * (values - range_max) ** 2
        ctx.save_for_backward(values)
        ctx.range_min = range_min
        ctx.range_max = range_max
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        values = ctx.saved_tensors[0]
        range_min, range_max = ctx.range_min, ctx.range_max
        zero = torch.zeros(1, device=values.device)
        dloss_dvalues = torch.heaviside(-(values - range_min), zero) * (values - range_min) * 2 \
                        + torch.heaviside(values - range_max, zero) * (values - range_max) * 2

        dL_dvalues = dL_dloss * dloss_dvalues

        return dL_dvalues, None, None


class _InterlevelLoss(torch.nn.Module):
    """
        from https://github.com/kwea123/ngp_pl/blob/495702f414082e993e3b40757caa08792ec9be25/losses.py#L26
    """

    def outer_loss(self, bins, weights, proposal_bins, proposal_weights):
        starts = bins[..., :-1]
        ends = bins[..., 1:]
        proposal_starts = proposal_bins[..., :-1]
        proposal_ends = proposal_bins[..., 1:]

        cumulative_proposal_weights = torch.cat([torch.zeros_like(proposal_weights[..., :1]),
                                                 torch.cumsum(proposal_weights, dim=-1)], dim=-1)

        idx_overlap_starts = torch.clamp(torch.searchsorted(proposal_starts.contiguous(), starts.contiguous(),
                                                            side="right") - 1, min=0, max=proposal_weights.size(-1) - 1)
        idx_overlap_ends = torch.clamp(torch.searchsorted(proposal_ends.contiguous(), ends.contiguous(),
                                                          side="right") - 1, min=0, max=proposal_weights.size(-1) - 1)
        cumulative_proposal_weights_of_starts = torch.gather(cumulative_proposal_weights[..., :-1], -1,
                                                             idx_overlap_starts)
        cumulative_proposal_weights_of_ends = torch.gather(cumulative_proposal_weights[..., 1:], -1,
                                                           idx_overlap_ends)
        outer_weights_sum = cumulative_proposal_weights_of_ends - cumulative_proposal_weights_of_starts

        return torch.clamp(weights - outer_weights_sum, min=0) ** 2 / (weights + 1e-7)

    def forward(self, proposal_bins_and_weights, prediction_bins_and_weights):
        pred_bins = prediction_bins_and_weights[0].detach()
        pred_weights = prediction_bins_and_weights[1].detach()
        loss = 0.0

        for proposal_bins, proposal_weights in proposal_bins_and_weights:
            loss += torch.mean(self.outer_loss(pred_bins, pred_weights, proposal_bins, proposal_weights))

        return loss


class CompositeLoss(torch.nn.Module):
    def __init__(self, losses):
        super(CompositeLoss, self).__init__()
        self.losses = losses
        self.config = [loss.config for loss in self.losses]

    def forward(self, renders, target_renders, render_infos):
        losses_dict = {}
        for loss in self.losses:
            loss_dict = loss(renders, target_renders, render_infos)
            losses_dict.update(loss_dict)

        return losses_dict

    def get_callbacks(self):
        callbacks = []
        for loss in self.losses:
            callbacks += loss.get_callbacks()

        return callbacks

    def save(self, path):
        save_path = os.path.join(path, "loss_config.json")
        with open(save_path, "w") as file:
            json.dump(self.config, file)


class BaseLoss(torch.nn.Module):
    def __init__(self, weight=1e-3, start_step=None, end_step=None, end_weight=None):
        super(BaseLoss, self).__init__()
        self.config = {k: v for k, v in vars().items() if k != "self" and k != "__class__"}
        self.config["class"] = type(self).__name__
        self.weight = weight
        self.start_step = start_step
        self.start_weight = weight
        self.end_step = end_step
        self.end_weight = end_weight
        self.global_step = 0

    def forward(self, render: Render, target_render: TargetRender, render_info: RenderInfo):
        raise NotImplementedError

    def get_callbacks(self):
        class UpdateCallback(Callback):
            def __init__(self, loss):
                self.loss = loss

            def on_step_begin(self, global_step):
                self.loss.global_step = global_step
                if self.loss.start_step is not None:
                    percentage = np.clip((global_step - self.loss.start_step) /
                                         max((self.loss.end_step - self.loss.start_step), 1e-8), 0, 1)
                    self.loss.weight = self.loss.start_weight + percentage * (self.loss.end_weight
                                                                              - self.loss.start_weight)

        return [UpdateCallback(self)]

    def save(self, path):
        save_path = os.path.join(path, "loss_config.json")
        with open(save_path, "w") as file:
            json.dump(self.config, file)


class RGBLossMSE(BaseLoss):

    def __init__(self, *args, weight=1.0, **kwargs):
        super(RGBLossMSE, self).__init__(*args, weight=weight, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})

    def forward(self, renders, target_renders, render_infos):
        rgb_losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            rgb_loss = (render.features - target_render.rgbs) ** 2
            if target_render.valid_mask is not None:
                rgb_loss = rgb_loss * target_render.valid_mask
            rgb_losses.append(rgb_loss.mean())

        final_loss = sum(rgb_losses) / len(rgb_losses)

        return {"rgb_loss_mse": self.weight * final_loss}


class RGBLossL1(BaseLoss):
    def __init__(self, *args, weight=1.0, **kwargs):
        super(RGBLossL1, self).__init__(*args, weight=weight, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})

    def forward(self, renders, target_renders, render_infos):
        rgb_losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            rgb_loss = torch.abs(render.features - target_render.rgbs)
            if target_render.valid_mask is not None:
                rgb_loss = rgb_loss * target_render.valid_mask
            rgb_losses.append(rgb_loss.mean())

        final_loss = sum(rgb_losses) / len(rgb_losses)

        return {"rgb_loss_l1": self.weight * final_loss}


class RGBLossSSIM(BaseLoss):
    def __init__(self, *args, weight=0.2, **kwargs):
        super().__init__(*args, weight=weight, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})

    def forward(self, renders, target_renders, render_infos):
        ssim_losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            rgb_render = render.features
            rgb_target = target_render.rgbs
            if target_render.valid_mask is not None:
                rgb_render = rgb_render * target_render.valid_mask
                rgb_target = rgb_target * target_render.valid_mask
            ssim_loss = 1 - structural_similarity_index_measure(
                rgb_render.reshape([1, int(render_info.height), int(render_info.width), -1]).moveaxis(-1, 1),
                rgb_target.reshape([1, int(render_info.height), int(render_info.width), -1]).moveaxis(-1, 1),
                data_range=1.0)
            ssim_losses.append(ssim_loss.mean())

        final_loss = sum(ssim_losses) / len(ssim_losses)
        return {"ssim_loss": self.weight * final_loss}


class RGBLossMSSSIM(BaseLoss):
    def __init__(self, *args, weight=0.2, **kwargs):
        super().__init__(*args, weight=weight, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})

    def forward(self, renders, target_renders, render_infos):
        ms_ssim_losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            rgb_render = render.features
            rgb_target = target_render.rgbs
            if target_render.valid_mask is not None:
                rgb_render = rgb_render * target_render.valid_mask
                rgb_target = rgb_target * target_render.valid_mask
            ms_ssim_loss = 1 - multiscale_structural_similarity_index_measure(
                rgb_render.reshape([1, int(render_info.height), int(render_info.width), -1]).moveaxis(-1, 0),
                rgb_target.reshape([1, int(render_info.height), int(render_info.width), -1]).moveaxis(-1, 0),
                data_range=1.0)
            ms_ssim_losses.append(ms_ssim_loss.mean())

        final_loss = sum(ms_ssim_losses) / len(ms_ssim_losses)

        return {"ms_ssim_loss": self.weight * final_loss}


class DepthLossMSE(BaseLoss):

    def __init__(self, *args, weight=1.0, **kwargs):
        super(DepthLossMSE, self).__init__(*args, weight=weight, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})

    def forward(self, renders, target_renders, render_infos):
        depth_losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            depth_loss = ((render.depths - target_render.depths) ** 2
                          )
            if target_render.valid_mask is not None:
                depth_loss = depth_loss * target_render.valid_mask.any(dim=-1)

            depth_losses.append(depth_loss.mean())

        final_loss = sum(depth_losses) / len(depth_losses)

        return {"depth_loss": self.weight * final_loss}


class GSDepthRankingLoss(BaseLoss):

    def __init__(self, *args, weight=0.2, continuity_weight=0.1, ranking_margin=1e-4, continuity_margin=1e-4,
                 sample_ratio=0.25, **kwargs):
        super(GSDepthRankingLoss, self).__init__(*args, weight=weight, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})

        self.ranking_margin = ranking_margin
        self.continuity_margin = continuity_margin
        self.continuity_weight = continuity_weight
        self.sample_ratio = sample_ratio

    def forward(self, renders, target_renders, render_infos):
        ranking_losses = []
        continuity_losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            depth_sample_idx = sample_depth_loss_indices(target_render.depths, render_info.width, render_info.height,
                                                         target_render.valid_mask,
                                                         n_samples=int(render_info.width * render_info.height
                                                                       * self.sample_ratio),
                                                         rank_patch_size=render_info.width // 8,
                                                         knn_crop_radius=3)
            depth_samples = render.depths[depth_sample_idx].reshape(-1, 2, 2)
            ranking_loss = torch.maximum(depth_samples[:, 0, 0] - depth_samples[:, 0, 1] + self.ranking_margin,
                                         torch.tensor(0))
            continuity_loss = torch.maximum(torch.abs(depth_samples[:, 0, :] - depth_samples[:, 1, :])
                                            - self.continuity_margin,
                                            torch.tensor(0))

            ranking_losses.append(ranking_loss.mean())
            continuity_losses.append(continuity_loss.mean())

        final_ranking_loss = sum(ranking_losses) / len(ranking_losses)
        final_continuity_loss = sum(continuity_losses) / len(continuity_losses)

        return {"depth_ranking_loss": self.weight * final_ranking_loss,
                "depth_continuity_loss": self.weight * self.continuity_weight * final_continuity_loss}


class PearsonLoss(BaseLoss):

    def __init__(self, *args, weight=1.0, **kwargs):
        super(PearsonLoss, self).__init__(*args, weight=weight, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})

    def forward(self, renders, target_renders, render_infos):
        pearson_losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            pearson_loss = torch.min(1 - pearson_corrcoef(render.depths, 1 / (10 * target_render.depths + 1)),
                                     1 - pearson_corrcoef(render.depths, -target_render.depths))

            pearson_losses.append(pearson_loss.mean())

        final_loss = sum(pearson_losses) / len(pearson_losses)

        return {"pearson_loss": self.weight * final_loss}


class PatchPearsonLoss(BaseLoss):

    def __init__(self, *args, weight=1.0, patch_size=128, **kwargs):
        super(PatchPearsonLoss, self).__init__(*args, weight=weight, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})
        self.patch_size = patch_size

    def forward(self, renders, target_renders, render_infos):
        patch_pearson_losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            render_depth = render.depths.reshape(int(render_info.height), int(render_info.width))
            target_render_depth = 1 / (10 * target_render.depths.reshape(int(render_info.height),
                                                                         int(render_info.width)) + 1)

            n_patches = torch.tensor([int(render_info.height), int(render_info.width)],
                                     dtype=torch.int) // self.patch_size
            max_offsets = torch.tensor([int(render_info.height), int(render_info.width)]) % self.patch_size

            n_patches = torch.minimum(n_patches, torch.tensor([1, 1]))
            height_offset = 0
            if max_offsets[0] != 0:
                height_offset = torch.randint(0, int(max_offsets[0]), (1,))
            width_offset = 0
            if max_offsets[1] != 0:
                width_offset = torch.randint(0, int(max_offsets[1]), (1,))

            croped_render_depth = render_depth[height_offset:height_offset + n_patches[0] * self.patch_size,
                                  width_offset:width_offset + n_patches[1] * self.patch_size]
            croped_target_render_depth = target_render_depth[
                                         height_offset:height_offset + n_patches[0] * self.patch_size,
                                         width_offset:width_offset + n_patches[1] * self.patch_size]

            patched_render_depth = croped_render_depth.unfold(0, self.patch_size, self.patch_size
                                                              ).unfold(1, self.patch_size, self.patch_size
                                                                       ).reshape(-1, self.patch_size, self.patch_size)

            patched_target_render_depth = croped_target_render_depth.unfold(0, self.patch_size, self.patch_size
                                                                            ).unfold(1, self.patch_size, self.patch_size
                                                                                     ).reshape(-1, self.patch_size,
                                                                                               self.patch_size)

            render_depth_centered = patched_render_depth - patched_render_depth.mean(dim=(-2, -1), keepdim=True)
            target_render_depth_centered = patched_target_render_depth - patched_target_render_depth.mean(dim=(-2, -1),
                                                                                                          keepdim=True)

            render_depth_norm = render_depth_centered / render_depth_centered.std(dim=(-2, -1), keepdim=True)
            target_render_depth_norm = target_render_depth_centered / target_render_depth_centered.std(dim=(-2, -1),
                                                                                                       keepdim=True)

            pearson_coefficient = (render_depth_norm * target_render_depth_norm).mean(dim=(-2, -1))

            pearson_loss = 1 - pearson_coefficient

            patch_pearson_losses.append(pearson_loss.mean())

        final_loss = sum(patch_pearson_losses) / len(patch_pearson_losses)

        return {"patch_pearson_loss": self.weight * final_loss}


class GSSparsityLoss(BaseLoss):
    def __init__(self, *args, weight=1e-6, **kwargs):
        super().__init__(*args, weight=weight, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})

    def forward(self, renders, target_renders, render_infos):
        losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            opacities = torch.abs(render.auxiliary.gaussian_values.opacities)

            losses.append(opacities.mean())

        final_loss = sum(losses) / len(losses)

        return {"gs_sparsity_loss": self.weight * final_loss}


class GSScaleLoss(BaseLoss):
    def __init__(self, *args, weight=0.01, **kwargs):
        super().__init__(*args, weight=weight, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})

    def forward(self, renders, target_renders, render_infos):
        losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            scales = torch.abs(render.auxiliary.gaussian_values.scales)

            losses.append(scales.mean())

        final_loss = sum(losses) / len(losses)

        return {"gs_scale_loss": self.weight * final_loss}


class GSFlatnessLoss(BaseLoss):
    def __init__(self, *args, weight=100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})

    def forward(self, renders, target_renders, render_infos):
        flatness_losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            scales = torch.abs(render.auxiliary.gaussian_values.scales)
            min_scale = torch.min(scales, dim=-1)[0]

            flatness_losses.append(min_scale.mean())

        final_loss = sum(flatness_losses) / len(flatness_losses)

        return {"gs_flatness_loss": self.weight * final_loss}


class GSOpacityLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})

    def forward(self, renders, target_renders, render_infos):
        opacity_losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            opacities = render.auxiliary.gaussian_values.opacities + 1e-10
            opacity_loss = self.weight * ((-opacities * torch.log(opacities)) +
                                          (-(1 - opacities)) * torch.log(1 - opacities))

            opacity_losses.append(opacity_loss.mean())

        final_loss = sum(opacity_losses) / len(opacity_losses)

        return {"gs_opacity_loss": final_loss}


class GSNormalLoss(BaseLoss):
    def __init__(self, *args, edge_aware=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})
        self.edge_aware = edge_aware

    def forward(self, renders, target_renders, render_infos):
        if self.weight <= 0:
            return {"gs_normal_loss": torch.tensor(0.0)}

        normal_losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            normals = render.normals
            depth_normals = render.auxiliary.depth_normals

            normal_loss = (1 - torch.sum(depth_normals * normals, dim=-1))

            if self.edge_aware:
                grad_weight = (1 - relative_image_gradient_magnitude(
                    target_render.rgbs.reshape([int(render_info.height),
                                                int(render_info.width),
                                                -1]))
                               ) ** 2
                # grad_weight = minpool_grayscale(grad_weight)
                normal_loss = normal_loss * grad_weight.reshape(-1)

            normal_losses.append(normal_loss.mean())

        final_loss = sum(normal_losses) / len(normal_losses)

        return {"gs_normal_loss": self.weight * final_loss}


class GSDistortionLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})

    def forward(self, renders, target_renders, render_infos):
        distortion_losses = []
        for render, target_render, render_info in zip(renders, target_renders, render_infos):
            distortion_loss = render.auxiliary.distortion_map

            distortion_losses.append(distortion_loss.mean())

        final_loss = sum(distortion_losses) / len(distortion_losses)

        return {"gs_distortion_loss": self.weight * final_loss}
