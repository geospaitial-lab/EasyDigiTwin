import json
import os.path

import math
import torch
from pykdtree.kdtree import KDTree

from src.datasets.cameras import TrainableCameras
from src.scenes.postprocessors import NoOpPostProcessor
from src.utils.callback_utils import Callback
from src.utils.general_utils import inv_sigmoid, C_0
from src.utils.interfaces import GaussianValues, RenderInfo
from src.utils.rotation_utils import quaternion_to_rotmat
from src.utils.save_load_utils import load_from_ply, load_from_las, save_pointcloud


def covariance_from_scale_and_rotation(scales, rotations):
    rot_mat = quaternion_to_rotmat(rotations)
    cov_0 = torch.diag_embed(scales ** 2)

    return rot_mat @ cov_0 @ rot_mat.transpose(1, 2)


class GaussianModel(torch.nn.Module):
    def __init__(self, sh_degree, use_mip_filter=False, init_scale_modifier=1.0, init_opacity_modifier=0.1,
                 compute_filter_frequency=30000, device="cuda"):
        super(GaussianModel, self).__init__()
        self.config = {k: v for k, v in vars().items() if k != "self" and k != "__class__"}
        self.config["class"] = type(self).__name__

        self.means = torch.empty(0)
        self.sh_ks_diffuse = torch.empty(0)
        self.sh_ks_specular = torch.empty(0)
        self.opacities = torch.empty(0)
        self.scales = torch.empty(0)
        self.rotations = torch.empty(0)
        self.filter_3d_variances = torch.empty(0) if use_mip_filter else None
        self.compute_filter_frequency = compute_filter_frequency
        self.cameras = None

        self.sh_degree = sh_degree

        self.device = device

        self.init_scale_modifier = init_scale_modifier
        self.init_opacity_modifier = init_opacity_modifier

    @classmethod
    def from_initial_points(cls, point_xyzs, point_rgbs=None, sh_degree=3, **kwargs):
        gaussians = cls(sh_degree, **kwargs)
        n_points = point_xyzs.shape[0]

        kd_tree = KDTree(point_xyzs.cpu().numpy())
        distances, _ = kd_tree.query(point_xyzs.cpu().numpy(), k=4)
        distances = torch.tensor(distances[:, 1:] ** 2, device=gaussians.device).mean(dim=-1).sqrt()

        scales = torch.log(torch.clamp_min(distances * gaussians.init_scale_modifier, 1e-7))[..., None].repeat(1, 3)

        point_xyzs = point_xyzs.cuda()
        if point_rgbs is None:
            point_rgbs = torch.rand(n_points, 3, device=point_xyzs.device) * 0.001 + 0.5
        else:
            point_rgbs = point_rgbs.cuda()

        point_sh_ks_diffuse = (point_rgbs - 0.5) / C_0

        sh_ks_specular = torch.zeros((n_points, (sh_degree + 1) ** 2 - 1, 3), device=gaussians.device)

        opacities = inv_sigmoid(gaussians.init_opacity_modifier * torch.ones((n_points, 1), device=gaussians.device))

        rotations = torch.zeros((n_points, 4), device=gaussians.device)
        rotations[..., 0] = 1

        gaussians.means = torch.nn.Parameter(point_xyzs)
        gaussians.sh_ks_diffuse = torch.nn.Parameter(point_sh_ks_diffuse)
        gaussians.sh_ks_specular = torch.nn.Parameter(sh_ks_specular)
        gaussians.opacities = torch.nn.Parameter(opacities)
        gaussians.scales = torch.nn.Parameter(scales)
        gaussians.rotations = torch.nn.Parameter(rotations)

        if gaussians.filter_3d_variances is not None:
            gaussians.filter_3d_variances = torch.zeros_like(gaussians.scales)

        return gaussians

    @classmethod
    def from_random_points(cls, n_points, bounds, sh_degree=3, **kwargs):
        points_xyz = torch.rand(n_points, 3) * torch.tensor([bounds[3] - bounds[0],
                                                             bounds[4] - bounds[1],
                                                             bounds[5] - bounds[2]]) + torch.tensor(bounds[:3])

        return cls.from_initial_points(points_xyz, sh_degree=sh_degree, **kwargs)

    @classmethod
    def from_numpy(cls, means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular, **kwargs):
        sh_degree = int(math.sqrt(sh_ks_specular.shape[-2] + 1) - 1)

        gaussians = cls(sh_degree, **kwargs)
        gaussians.means = torch.nn.Parameter(torch.tensor(means,
                                                          dtype=torch.float32, device=gaussians.device).contiguous())
        gaussians.sh_ks_diffuse = torch.nn.Parameter(torch.tensor(sh_ks_diffuse, dtype=torch.float32,
                                                                  device=gaussians.device).reshape(-1, 3).contiguous())
        gaussians.sh_ks_specular = torch.nn.Parameter(torch.tensor(sh_ks_specular, dtype=torch.float32,
                                                                   device=gaussians.device).contiguous())
        gaussians.opacities = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float32,
                                                              device=gaussians.device).contiguous())
        gaussians.scales = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float32,
                                                           device=gaussians.device).contiguous())
        gaussians.rotations = torch.nn.Parameter(torch.tensor(rotations, dtype=torch.float32,
                                                              device=gaussians.device).contiguous())

        if gaussians.filter_3d_variances is not None:
            gaussians.filter_3d_variances = torch.zeros_like(gaussians.scales)

        return gaussians

    def get_postprocessor(self):
        return NoOpPostProcessor()

    def to_numpy(self):
        means = self.means.detach().cpu().numpy()
        sh_ks_diffuse = self.sh_ks_diffuse.detach().cpu().numpy()
        sh_ks_specular = self.sh_ks_specular.detach().cpu().numpy()
        opacities, scales = self.get_opacities_and_scales()
        opacities = inv_sigmoid(opacities).detach().cpu().numpy()
        scales = torch.log(scales).detach().cpu().numpy()
        rotations = self.rotations.detach().cpu().numpy()

        return means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular

    def save(self, base_path, config_path, model_path):
        self.config["model_path"] = model_path
        with open(os.path.join(base_path, config_path), "w") as file:
            json.dump(self.config, file)
        save_pointcloud(os.path.join(base_path, model_path), *self.to_numpy(), rgb_format="rgb")

    def get_sh_ks(self):
        sh_ks = torch.cat([self.sh_ks_diffuse[..., None, :], self.sh_ks_specular], dim=1)
        return sh_ks

    def get_opacities_and_scales(self):
        opacities = torch.nan_to_num(torch.sigmoid(self.opacities))
        scales = torch.exp(self.scales)

        if self.filter_3d_variances is None:
            return opacities, scales

        variances = scales ** 2
        determinant_unfiltered = torch.prod(variances, dim=-1, keepdim=True)
        filtered_variances = variances + self.filter_3d_variances
        determinant_filtered = torch.prod(filtered_variances, dim=-1, keepdim=True)
        normalization_coefficient = torch.sqrt(determinant_unfiltered / determinant_filtered)

        return opacities * normalization_coefficient, torch.sqrt(filtered_variances)

    def inv_get_opacities_and_scales(self, filtered_opacities, filtered_scales, idx=None):
        if self.filter_3d_variances is not None:
            filtered_variances = filtered_scales ** 2
            if idx is not None:
                filter_3d_variances = self.filter_3d_variances[idx]
            else:
                filter_3d_variances = self.filter_3d_variances
            variances = torch.clamp(filtered_variances - filter_3d_variances, min=1e-7)
            activated_scales = torch.sqrt(variances)
            determinant_unfiltered = torch.prod(variances, dim=-1, keepdim=True)
            determinant_filtered = torch.prod(filtered_variances, dim=-1, keepdim=True)
            normalization_coefficient = torch.sqrt(determinant_unfiltered / determinant_filtered)
            activated_opacities = filtered_opacities / normalization_coefficient
        else:
            activated_opacities = filtered_opacities
            activated_scales = filtered_scales

        opacities = inv_sigmoid(activated_opacities)
        scales = torch.log(activated_scales)

        return opacities, scales

    def get_covariances_and_opacities(self):
        opacities, scales = self.get_opacities_and_scales()
        rotations = torch.nn.functional.normalize(self.rotations)

        return covariance_from_scale_and_rotation(scales, rotations), opacities

    def forward(self):
        means = self.means

        sh_ks = self.get_sh_ks()
        opacities, scales = self.get_opacities_and_scales()
        rotations = torch.nn.functional.normalize(self.rotations)

        return GaussianValues(means, sh_ks, opacities, scales, rotations).contiguous()

    def set_cameras(self, cameras: TrainableCameras):
        self.cameras = [cameras]

    @torch.no_grad()
    def compute_filter_3d_variances(self):
        means = torch.cat([self.means, torch.ones_like(self.means[..., :1])], dim=-1)
        min_depth_in_cams = torch.ones_like(means[..., 0]) * 1e6
        point_is_seen = torch.zeros(means.shape[0], dtype=torch.bool, device=means.device)
        max_focal_length = 0.0
        cameras = self.cameras[0]

        for i in range(len(cameras)):
            render_info: RenderInfo = cameras([i], ..., return_rays=False).to(means.device)
            focal_length = render_info.intrinsics[0]
            max_focal_length = max(max_focal_length, focal_length)

            view_transform = render_info.view_transform
            means_view = torch.einsum("jk,nk->nj", view_transform, means)
            mean_screen = means_view[..., :2] / (means_view[..., 2:3] + 1e-8)

            in_frustum = torch.logical_and(
                torch.logical_and(
                    torch.logical_and(mean_screen[..., 0] >= -1.15 * render_info.width / (2 * focal_length),
                                      mean_screen[..., 0] <= 1.15 * render_info.width / (2 * focal_length)),
                    torch.logical_and(mean_screen[..., 1] >= -1.15 * render_info.height / (2 * focal_length),
                                      mean_screen[..., 1] <= 1.15 * render_info.height / (2 * focal_length))),
                means_view[..., 2] > 0.002)

            point_is_seen[in_frustum] = True
            min_depth_in_cams[in_frustum] = torch.minimum(min_depth_in_cams[in_frustum], means_view[in_frustum, 2])

        min_depth_in_cams[~point_is_seen] = min_depth_in_cams[point_is_seen].max()
        self.filter_3d_variances = ((min_depth_in_cams / max_focal_length) ** 2 * 0.2)[..., None]

    def get_callbacks(self):
        class UpdateCallback(Callback):
            def __init__(self, gaussian_model: GaussianModel):
                self.gaussian_model = gaussian_model

            def on_step_begin(self, global_step):
                if global_step % self.gaussian_model.compute_filter_frequency == 0:
                    if self.gaussian_model.filter_3d_variances is not None:
                        self.gaussian_model.compute_filter_3d_variances()

        return [UpdateCallback(self)]


def get_gaussian_model(classname):
    return eval(classname)


def load_gaussian_model(path, device="cuda", use_mip_filter=False):
    if path.endswith(".ply"):
        means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular = load_from_ply(path)
    elif path.endswith(".las") or path.endswith(".laz"):
        means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular, _, _ = load_from_las(path)
    else:
        raise ValueError(f"Unknown file format: {path}")

    model = GaussianModel.from_numpy(means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular,
                                    use_mip_filter=use_mip_filter).to(device)

    return model
