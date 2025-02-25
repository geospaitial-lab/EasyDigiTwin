import math
import torch

from src.utils.general_utils import inv_sigmoid
from src.utils.rotation_utils import quaternion_to_rotmat
from src.utils.callback_utils import Callback
from edt_gaussian_rasterization import compute_relocation


class GaussianDensifier:
    def __init__(self, model, densification_start=500, densification_end=15000, densification_frequency=100):
        self.densification_start = densification_start
        self.densification_end = densification_end
        self.densification_frequency = densification_frequency

        self.gaussians = model

    def resize_parameters_in_optimizer(self, optimizer, selection_mask, extension_dict):
        for param_group in optimizer.param_groups:
            assert len(param_group["params"]) == 1
            group_name = param_group["name"]
            group_params = param_group["params"][0]
            group_state = optimizer.state.get(group_params)

            extension_tensor = extension_dict.get(group_name)

            if extension_tensor is not None:
                del extension_dict[group_name]
                param_group["params"][0] = torch.nn.Parameter(torch.cat([group_params[selection_mask],
                                                                         extension_tensor], dim=0))

                if group_state is not None:
                    for state_key in group_state:
                        if group_state[state_key].shape == group_params.shape:
                            group_state[state_key] = torch.cat([group_state[state_key][selection_mask],
                                                                torch.zeros_like(extension_tensor)], dim=0)
                    del optimizer.state[group_params]
                    optimizer.state[param_group["params"][0]] = group_state

                self.gaussians.__setattr__(group_name, param_group["params"][0])

    def update_stats(self, means_2d, radii):
        # Called on every render
        pass

    def get_callbacks(self):
        return []


class ScreenGradientDensifier(GaussianDensifier):
    def __init__(self, model, densification_start=500, densification_end=15000, densification_frequency=100,
                 opacity_reduction_frequency=3000, clone_split_threshold=0.01, densification_grad_threshold=0.0002,
                 prune_opacity_threshold=0.005, use_abs=True):
        super().__init__(model, densification_start, densification_end, densification_frequency)

        self.opacity_reduction_frequency = opacity_reduction_frequency
        self.clone_split_threshold = clone_split_threshold
        self.densification_grad_threshold = densification_grad_threshold
        self.prune_opacity_threshold = prune_opacity_threshold

        self.viewspace_gradient_norms_accumulated = torch.empty(0)
        self.viewspace_gradient_abs_accumulated = torch.empty(0)
        self.viewspace_gradient_denominators = torch.empty(0)

        self.last_viewspace_means = None
        self.last_viewspace_accumulation_mask = None
        self.use_abs = use_abs

        self.rebuild()

    def rebuild(self):
        self.viewspace_gradient_norms_accumulated = torch.zeros_like(self.gaussians.means[..., 0])
        self.viewspace_gradient_abs_accumulated = torch.zeros_like(self.gaussians.means[..., 0])
        self.viewspace_gradient_denominators = torch.zeros_like(self.gaussians.means[..., 0])

    def update_stats(self, means_2d, radii):
        mask = radii > 0
        self.last_viewspace_means = means_2d
        self.last_viewspace_accumulation_mask = mask

    def accumulate_viewspace_gradients(self, grad_scale=1.0):
        viewspace_grads = self.last_viewspace_means.grad / grad_scale
        accumulation_mask = self.last_viewspace_accumulation_mask

        if viewspace_grads is None:
            return
        self.viewspace_gradient_norms_accumulated[accumulation_mask] += (
            torch.norm(viewspace_grads[accumulation_mask, :2], dim=-1))
        self.viewspace_gradient_abs_accumulated[accumulation_mask] += (
            torch.norm(viewspace_grads[accumulation_mask, 2:], dim=-1))
        self.viewspace_gradient_denominators[accumulation_mask] += 1

        self.last_viewspace_means = None
        self.last_viewspace_accumulation_mask = None

    @torch.no_grad()
    def set_opacity(self, new_opacity=0.01):
        new_opacity = torch.tensor(new_opacity,
                                   dtype=self.gaussians.opacities.dtype,
                                   device=self.gaussians.opacities.device)
        new_opacities = torch.min(self.gaussians.opacities,
                                  torch.ones_like(self.gaussians.opacities) * inv_sigmoid(new_opacity))
        self.gaussians.opacities.copy_(new_opacities)

    def get_cloned_gaussians(self, selection_mask):
        cloned_means = self.gaussians.means[selection_mask]
        cloned_rgbs = self.gaussians.sh_ks_diffuse[selection_mask]
        cloned_sh_ks_specular = self.gaussians.sh_ks_specular[selection_mask]
        cloned_opacities = self.gaussians.opacities[selection_mask]
        cloned_scales = self.gaussians.scales[selection_mask]
        cloned_rotations = self.gaussians.rotations[selection_mask]
        cloned_filter_3d_variances = self.gaussians.filter_3d_variances[
            selection_mask] if self.gaussians.filter_3d_variances is not None else None

        return (cloned_means, cloned_rgbs, cloned_sh_ks_specular, cloned_opacities, cloned_scales, cloned_rotations,
                cloned_filter_3d_variances)

    def get_split_gaussians(self, selection_mask):

        split_sample_stds = torch.exp(self.gaussians.scales)[selection_mask].repeat(2, 1)
        split_sample_means = torch.zeros_like(split_sample_stds)
        split_samples = torch.normal(split_sample_means, split_sample_stds)
        split_sample_rotation_mats = quaternion_to_rotmat(torch.nn.functional.normalize(
            self.gaussians.rotations)[selection_mask]).repeat(2, 1, 1)
        split_means = (torch.einsum("njk,nk->nj", split_sample_rotation_mats, split_samples)
                       + self.gaussians.means[selection_mask].repeat(2, 1))
        split_rgbs = self.gaussians.sh_ks_diffuse[selection_mask].repeat(2, 1)
        split_sh_ks_specular = self.gaussians.sh_ks_specular[selection_mask].repeat(2, 1, 1)
        split_opacities = self.gaussians.opacities[selection_mask].repeat(2, 1)
        split_scales = torch.log(split_sample_stds / 1.6)
        split_rotations = self.gaussians.rotations[selection_mask].repeat(2, 1)
        split_filter_3d_variances = self.gaussians.filter_3d_variances[selection_mask].repeat(2, 1) \
            if self.gaussians.filter_3d_variances is not None else None

        return (split_means, split_rgbs, split_sh_ks_specular, split_opacities, split_scales, split_rotations,
                split_filter_3d_variances)

    @torch.no_grad()
    def split_clone_remove(self, optimizers):
        accumulated_gradients_mean = (self.viewspace_gradient_norms_accumulated /
                                      (self.viewspace_gradient_denominators + 1e-8))
        accumulated_gradients_abs = (self.viewspace_gradient_abs_accumulated /
                                     (self.viewspace_gradient_denominators + 1e-8))

        densification_selection = accumulated_gradients_mean >= self.densification_grad_threshold
        densification_selection_abs = accumulated_gradients_abs >= self.densification_grad_threshold * 2

        large_gaussian_selection = torch.max(torch.exp(self.gaussians.scales), dim=-1)[0] >= self.clone_split_threshold
        if self.use_abs:
            split_selection = torch.logical_and(densification_selection_abs, large_gaussian_selection)
        else:
            split_selection = torch.logical_and(densification_selection, large_gaussian_selection)
        clone_selection = torch.logical_and(densification_selection, torch.logical_not(large_gaussian_selection))
        opacity_selection = (torch.sigmoid(self.gaussians.opacities) < self.prune_opacity_threshold).squeeze()
        too_big_selection = torch.max(torch.exp(self.gaussians.scales), dim=-1)[0] > 0.1
        prune_selection = torch.logical_or(opacity_selection, too_big_selection)
        keep_selection = ~torch.logical_or(split_selection, prune_selection)

        (cloned_means, cloned_rgbs, cloned_sh_ks_specular,
         cloned_opacities, cloned_scales, cloned_rotations,
         cloned_filter_3d_variances) = self.get_cloned_gaussians(clone_selection)

        (split_means, split_rgbs, split_sh_ks_specular,
         split_opacities, split_scales, split_rotations,
         split_filter_3d_variances) = self.get_split_gaussians(split_selection)

        extension_dict = {"means": torch.cat([cloned_means, split_means], dim=0),
                          "sh_ks_diffuse": torch.cat([cloned_rgbs, split_rgbs], dim=0),
                          "sh_ks_specular": torch.cat([cloned_sh_ks_specular, split_sh_ks_specular], dim=0),
                          "opacities": torch.cat([cloned_opacities, split_opacities], dim=0),
                          "scales": torch.cat([cloned_scales, split_scales], dim=0),
                          "rotations": torch.cat([cloned_rotations, split_rotations], dim=0),
                          }

        if self.gaussians.filter_3d_variances is not None:
            self.gaussians.filter_3d_variances = torch.cat([self.gaussians.filter_3d_variances[keep_selection],
                                                            cloned_filter_3d_variances,
                                                            split_filter_3d_variances], dim=0)

        for optimizer in optimizers.values():
            if "means" in [param_group.get("name") for param_group in optimizer.param_groups]:
                self.resize_parameters_in_optimizer(optimizer, keep_selection, extension_dict)

        assert len(extension_dict) == 0

        self.rebuild()
        torch.cuda.empty_cache()

    def get_callbacks(self):
        class UpdateCallback(Callback):
            def __init__(self, gaussian_densification: ScreenGradientDensifier):
                self.gaussian_densification = gaussian_densification

            def on_post_optimizers_step(self, global_step, optimizers):
                if global_step < self.gaussian_densification.densification_end:
                    if (global_step >= self.gaussian_densification.densification_start
                            and global_step % self.gaussian_densification.densification_frequency == 0):
                        self.gaussian_densification.split_clone_remove(optimizers)

                    if global_step % self.gaussian_densification.opacity_reduction_frequency == 0 and global_step != 0:
                        self.gaussian_densification.set_opacity()

            def on_backward(self, global_step, grad_scale=1.0):
                if global_step < self.gaussian_densification.densification_end:
                    self.gaussian_densification.accumulate_viewspace_gradients(grad_scale)

        return [UpdateCallback(self)]


class StreetScreenGradientDensifier(ScreenGradientDensifier):
    def __init__(self, *args,
                 clone_split_threshold=0.0001,
                 densification_grad_threshold=0.015,
                 densification_frequency=300,
                 use_abs=False,
                 **kwargs):
        super(StreetScreenGradientDensifier, self).__init__(*args,
                                                            clone_split_threshold=clone_split_threshold,
                                                            densification_grad_threshold=densification_grad_threshold,
                                                            densification_frequency=densification_frequency,
                                                            use_abs=use_abs,
                                                            **kwargs)
        self.viewspace_gradient_max = torch.empty(0)
        self.max_screen_size = torch.empty(0)

        self.rebuild()

    def rebuild(self):
        self.viewspace_gradient_max = torch.zeros_like(self.gaussians.means[..., 0])
        self.max_screen_size = torch.zeros_like(self.gaussians.means[..., 0])

    def update_stats(self, means_2d, radii):
        mask = radii > 0
        self.last_viewspace_means = means_2d
        self.last_viewspace_accumulation_mask = mask
        self.max_screen_size[mask] = torch.max(radii[mask], self.max_screen_size[mask])

    def accumulate_viewspace_gradients(self, grad_scale=1.0):
        viewspace_grads = self.last_viewspace_means.grad / grad_scale
        accumulation_mask = self.last_viewspace_accumulation_mask

        if viewspace_grads is None:
            return
        if self.use_abs:
            self.viewspace_gradient_max[accumulation_mask] = torch.max(
                torch.norm(viewspace_grads[accumulation_mask, 2:], dim=-1),
                self.viewspace_gradient_max[accumulation_mask])
        else:
            self.viewspace_gradient_max[accumulation_mask] = torch.max(
                torch.norm(viewspace_grads[accumulation_mask, :2], dim=-1),
                self.viewspace_gradient_max[accumulation_mask])

        self.last_viewspace_means = None
        self.last_viewspace_accumulation_mask = None

    @torch.no_grad()
    def split_clone_remove(self, optimizers):
        max_gradients = self.viewspace_gradient_max

        densification_selection = (max_gradients *
                                   self.max_screen_size *
                                   (torch.sigmoid(self.gaussians.opacities).squeeze() ** 1/5)
                                   >= self.densification_grad_threshold)
        densification_selection = torch.logical_and(densification_selection,
                                                    torch.sigmoid(self.gaussians.opacities).squeeze() >= 0.15)

        large_gaussian_selection = torch.max(torch.exp(self.gaussians.scales), dim=-1)[0] >= self.clone_split_threshold
        split_selection = torch.logical_and(densification_selection, large_gaussian_selection)
        clone_selection = torch.logical_and(densification_selection, torch.logical_not(large_gaussian_selection))

        prune_selection = (torch.sigmoid(self.gaussians.opacities) < self.prune_opacity_threshold).squeeze()
        keep_selection = ~torch.logical_or(split_selection, prune_selection)

        (cloned_means, cloned_rgbs, cloned_sh_ks_specular,
         cloned_opacities, cloned_scales, cloned_rotations,
         cloned_filter_3d_variances) = self.get_cloned_gaussians(clone_selection)

        (split_means, split_rgbs, split_sh_ks_specular,
         split_opacities, split_scales, split_rotations,
         split_filter_3d_variances) = self.get_split_gaussians(split_selection)

        extension_dict = {"means": torch.cat([cloned_means, split_means], dim=0),
                          "sh_ks_diffuse": torch.cat([cloned_rgbs, split_rgbs], dim=0),
                          "sh_ks_specular": torch.cat([cloned_sh_ks_specular, split_sh_ks_specular], dim=0),
                          "opacities": torch.cat([cloned_opacities, split_opacities], dim=0),
                          "scales": torch.cat([cloned_scales, split_scales], dim=0),
                          "rotations": torch.cat([cloned_rotations, split_rotations], dim=0),
                          }

        if self.gaussians.filter_3d_variances is not None:
            self.gaussians.filter_3d_variances = torch.cat([self.gaussians.filter_3d_variances[keep_selection],
                                                            cloned_filter_3d_variances,
                                                            split_filter_3d_variances], dim=0)

        for optimizer in optimizers.values():
            if "means" in [param_group.get("name") for param_group in optimizer.param_groups]:
                self.resize_parameters_in_optimizer(optimizer, keep_selection, extension_dict)

        assert len(extension_dict) == 0

        self.rebuild()
        torch.cuda.empty_cache()


class MCMCDensifier(GaussianDensifier):
    def __init__(self, model, densification_start=500, densification_end=25000, densification_frequency=100,
                 prune_opacity_threshold=0.005, growth_rate=1.05, max_n_gaussians=4.2e6, max_assigned=51,
                 sigmoid_steepness=100, noise_magnitude=5e5):
        super().__init__(model, densification_start, densification_end, densification_frequency)
        self.prune_opacity_threshold = prune_opacity_threshold
        self.growth_rate = growth_rate
        self.max_n_gaussians = int(max_n_gaussians)

        self.max_assigned = max_assigned
        self.binoms = torch.zeros((self.max_assigned, self.max_assigned)).float().cuda()
        for n in range(self.max_assigned):
            for k in range(n + 1):
                self.binoms[n, k] = math.comb(n, k)

        self.sigmoid_steepness = sigmoid_steepness
        self.noise_magnitude = noise_magnitude

    def compute_relocation_cuda(self, opacity_old, scale_old, n_assigned):
        n_assigned.clamp_(min=1, max=self.max_assigned - 1)
        return compute_relocation(opacity_old, scale_old, n_assigned, self.binoms, self.max_assigned)

    def reset_state_in_optimizer(self, optimizer, idx):
        for param_group in optimizer.param_groups:
            assert len(param_group["params"]) == 1
            group_params = param_group["params"][0]
            group_state = optimizer.state.get(group_params)

            if group_state is not None:
                for state_key in group_state:
                    if group_state[state_key].shape == group_params.shape:
                        group_state[state_key][idx] = 0
                del optimizer.state[group_params]
                optimizer.state[param_group["params"][0]] = group_state

    @torch.no_grad()
    def add_and_respawn(self, optimizers):
        n_current = self.gaussians.means.shape[0]
        n_added = max(0, min(self.max_n_gaussians, int(self.growth_rate * n_current)) - n_current)

        if n_added > 0:

            extension_dict = {"means": self.gaussians.means[:n_added],
                              "sh_ks_diffuse": self.gaussians.sh_ks_diffuse[:n_added],
                              "sh_ks_specular": self.gaussians.sh_ks_specular[:n_added],
                              "opacities": torch.zeros_like(self.gaussians.opacities[:n_added]),
                              "scales": self.gaussians.scales[:n_added],
                              "rotations": self.gaussians.rotations[:n_added],
                              }
            if self.gaussians.filter_3d_variances is not None:
                self.gaussians.filter_3d_variances = torch.cat([self.gaussians.filter_3d_variances,
                                                                torch.zeros_like(
                                                                    self.gaussians.filter_3d_variances[:n_added])],
                                                               dim=0)

            for optimizer in optimizers.values():
                if "means" in [param_group.get("name") for param_group in optimizer.param_groups]:
                    keep_selection = torch.ones_like(self.gaussians.means[..., 0], dtype=torch.bool)
                    self.resize_parameters_in_optimizer(optimizer, keep_selection, extension_dict)

            # torch.cuda.empty_cache()

        dead_gaussians_mask = (torch.sigmoid(self.gaussians.opacities) < self.prune_opacity_threshold).squeeze()
        n_dead = dead_gaussians_mask.sum()
        if n_dead == 0:
            return
        dead_gaussians_idx = torch.where(dead_gaussians_mask)[0]
        alive_gaussians_idx = torch.where(~dead_gaussians_mask)[0]

        opacities, scales = self.gaussians.get_opacities_and_scales()

        alive_opacities = opacities[alive_gaussians_idx].squeeze()
        gaussian_probabilities = alive_opacities / (alive_opacities.sum() + 1e-7)

        target_idx = torch.multinomial(gaussian_probabilities, n_dead, replacement=True)
        target_idx = alive_gaussians_idx[target_idx]
        n_assigned = torch.bincount(target_idx)[target_idx] + 1  # +1 because target itself must be included

        new_opacity, new_scale = self.compute_relocation_cuda(opacities[target_idx], scales[target_idx], n_assigned)
        new_opacity = torch.clamp(new_opacity, self.prune_opacity_threshold, 1 - 1e-7).reshape(-1, opacities.shape[-1])
        new_scale = new_scale.reshape(-1, scales.shape[-1])
        new_opacity, new_scale = self.gaussians.inv_get_opacities_and_scales(new_opacity, new_scale, idx=target_idx)

        self.gaussians.means[dead_gaussians_idx] = self.gaussians.means[target_idx]
        self.gaussians.sh_ks_diffuse[dead_gaussians_idx] = self.gaussians.sh_ks_diffuse[target_idx]
        self.gaussians.sh_ks_specular[dead_gaussians_idx] = self.gaussians.sh_ks_specular[target_idx]
        self.gaussians.opacities[dead_gaussians_idx] = new_opacity
        self.gaussians.scales[dead_gaussians_idx] = new_scale
        self.gaussians.rotations[dead_gaussians_idx] = self.gaussians.rotations[target_idx]
        if self.gaussians.filter_3d_variances is not None:
            self.gaussians.filter_3d_variances[dead_gaussians_idx] = self.gaussians.filter_3d_variances[target_idx]

        self.gaussians.opacities[target_idx] = new_opacity
        self.gaussians.scales[target_idx] = new_scale

        for optimizer in optimizers.values():
            if "means" in [param_group.get("name") for param_group in optimizer.param_groups]:
                self.reset_state_in_optimizer(optimizer, target_idx)

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    def _update_params(self, idxs, ratio, opacity_old, scale_old):
        new_opacity, new_scaling = self.compute_relocation_cuda(
            opacity_old=opacity_old[idxs, 0],
            scale_old=scale_old[idxs],
            n_assigned=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = inv_sigmoid(new_opacity)
        new_scaling = torch.log(new_scaling.reshape(-1, 3))

        return (self.gaussians.means[idxs], self.gaussians.sh_ks_diffuse[idxs], self.gaussians.sh_ks_specular[idxs],
                new_opacity, new_scaling, self.gaussians.rotations[idxs])

    @torch.no_grad()
    def relocate_gs(self, optimizers):
        dead_mask = (torch.sigmoid(self.gaussians.opacities) < self.prune_opacity_threshold).squeeze()

        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        opacities, scales = self.gaussians.get_opacities_and_scales()

        # sample from alive ones based on opacity
        probs = (opacities[alive_indices, 0])
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self.gaussians.means[dead_indices],
            self.gaussians.sh_ks_diffuse[dead_indices],
            self.gaussians.sh_ks_specular[dead_indices],
            self.gaussians.opacities[dead_indices],
            self.gaussians.scales[dead_indices],
            self.gaussians.rotations[dead_indices]
        ) = self._update_params(reinit_idx, ratio=ratio, opacity_old=opacities, scale_old=scales)

        self.gaussians.opacities[reinit_idx] = self.gaussians.opacities[dead_indices]
        self.gaussians.scales[reinit_idx] = self.gaussians.scales[dead_indices]

        for optimizer in optimizers.values():
            if "means" in [param_group.get("name") for param_group in optimizer.param_groups]:
                self.reset_state_in_optimizer(optimizer, reinit_idx)

    @torch.no_grad()
    def add_new_gs(self, optimizers):
        current_num_points = self.gaussians.opacities.shape[0]
        target_num = min(self.max_n_gaussians, int(self.growth_rate * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        opacities, scales = self.gaussians.get_opacities_and_scales()

        probs = opacities.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_means,
            new_sh_ks_diffuse,
            new_sh_ks_specular,
            new_opacities,
            new_scales,
            new_rotations
        ) = self._update_params(add_idx, ratio=ratio, opacity_old=opacities, scale_old=scales)

        self.gaussians.opacities[add_idx] = new_opacities
        self.gaussians.scales[add_idx] = new_scales

        extension_dict = {"means": new_means,
                          "sh_ks_diffuse": new_sh_ks_diffuse,
                          "sh_ks_specular": new_sh_ks_specular,
                          "opacities": new_opacities,
                          "scales": new_scales,
                          "rotations": new_rotations,
                          }

        for optimizer in optimizers.values():
            if "means" in [param_group.get("name") for param_group in optimizer.param_groups]:
                self.reset_state_in_optimizer(optimizer, add_idx)

                keep_selection = torch.ones_like(self.gaussians.means[..., 0], dtype=torch.bool)
                self.resize_parameters_in_optimizer(optimizer, keep_selection, extension_dict)

        return num_gs

    def noise_sigmoid(self, x):
        return 1 / (1 + torch.exp(-self.sigmoid_steepness * (x - (1 - self.prune_opacity_threshold))))

    @torch.no_grad()
    def add_noise(self, lr):
        covariances, opacities = self.gaussians.get_covariances_and_opacities()
        noise = torch.rand_like(self.gaussians.means) * self.noise_sigmoid(1 - opacities) * self.noise_magnitude * lr
        noise = (covariances @ noise[..., None])[..., 0]
        self.gaussians.means += noise

    def get_callbacks(self):
        class UpdateCallback(Callback):
            def __init__(self, gaussian_densification: MCMCDensifier):
                self.gaussian_densification = gaussian_densification

            def on_pre_optimizers_step(self, global_step, optimizers):
                if global_step < self.gaussian_densification.densification_end:
                    if (global_step >= self.gaussian_densification.densification_start
                            and global_step % self.gaussian_densification.densification_frequency == 0):
                        self.gaussian_densification.add_and_respawn(optimizers)

            def on_post_optimizers_step(self, global_step, optimizers):
                means_lr = [[param_group["lr"] for param_group in optimizer.param_groups
                             if param_group.get("name") == "means"]
                            for optimizer in optimizers.values()][0][0]
                self.gaussian_densification.add_noise(means_lr)

        return [UpdateCallback(self)]
