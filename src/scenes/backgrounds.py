import os.path
from typing import Tuple

import imgui
import numpy as np
import torch
from einops import rearrange
from pykdtree.kdtree import KDTree

from src.scenes.gaussian.gaussian_renderers import GaussianRenderer
from src.utils.general_utils import inv_sigmoid
from src.utils.interfaces import Render, RenderInfo, RenderState, GaussianValues
from src.utils.save_load_utils import save_to_checkpoint, load_from_checkpoint


class BaseBackgroundModel(torch.nn.Module):
    def __init__(self):
        super(BaseBackgroundModel, self).__init__()
        self.config = {k: v for k, v in vars().items() if k != "self" and k != "__class__"}
        self.config["class"] = type(self).__name__
        self.device_tracker = torch.nn.Parameter(torch.empty(0))
        self.is_trainable = False
        self.show_constant_background = False
        self.constant_color = torch.tensor([1.0, 1.0, 1.0])

    def forward(self, render_info: RenderInfo) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def composit(self, render: Render, render_info: RenderInfo) -> Render:
        background_color, background_dict = self(render_info)
        render.features = render.features + background_color * rearrange(1 - render.opacities, 'n -> n 1')
        render.background_auxiliary = background_dict
        render.state = RenderState.COMPOSITED
        return render

    def get_callbacks(self):
        return []

    def save(self, path):
        save_path = os.path.join(path, "background.ckpt")
        save_to_checkpoint(self, save_path)

    def draw_ui(self):
        pass


def get_beckground_model(classname):
    return eval(classname)


def load_background(path, device="cpu"):
    if os.path.isdir(path):
        path = os.path.join(path, "background.ckpt")

    background = load_from_checkpoint(path, get_beckground_model, device=device)

    return background


class ConstantBackgroundModel(BaseBackgroundModel):
    def __init__(self, color):
        super(ConstantBackgroundModel, self).__init__()
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"})
        self.register_buffer("color", torch.tensor(color) / 255.0)

    def forward(self, render_info):
        return self.color, {}

    def draw_ui(self):
        with imgui.begin_menu("Background", True) as main_menu:
            if main_menu.opened:
                current_color = self.color.cpu().numpy()
                _, new_color = imgui.color_edit3("Color", current_color[0], current_color[1], current_color[2])
                self.color = torch.tensor(new_color, device=self.color.device)


class RandomBackgroundModel(BaseBackgroundModel):
    def forward(self, render_info: RenderInfo):
        if self.show_constant_background:
            return self.constant_color.to(self.device_tracker.device), {}
        else:
            return torch.rand(3, device=self.device_tracker.device), {}

    def draw_ui(self):
        with imgui.begin_menu("Background", True) as main_menu:
            if main_menu.opened:
                current_color = self.constant_color.cpu().numpy()
                _, new_color = imgui.color_edit3("Color", current_color[0], current_color[1], current_color[2])
                self.constant_color = torch.tensor(new_color, device=self.constant_color.device)


C_0 = 0.28209479177387814


class GSSkyboxBackgroundModel(BaseBackgroundModel):
    def __init__(self, n_gaussians=100_000, radius=10, sh_degree=3, sh_update_frequency=1000, sh_update_start=1000,
                 device="cuda"):
        super(GSSkyboxBackgroundModel, self).__init__()
        self.is_trainable = True
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"})

        self.n_gaussians = n_gaussians
        self.radius = radius

        self.sh_degree = sh_degree

        indices = np.arange(0, self.n_gaussians, dtype=float) + 0.5

        phi = np.arccos(indices / self.n_gaussians)
        theta = np.pi * (1 + 5 ** 0.5) * indices

        x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        points = np.stack([x, y, z], axis=1) * self.radius

        point_xyzs = torch.from_numpy(points.astype(np.float32)).to(device)

        point_rgbs = torch.rand(self.n_gaussians, 3, device=point_xyzs.device) * 0.001 + 0.5
        point_rgbs = point_rgbs.to(device)

        point_sh_ks_diffuse = (point_rgbs - 0.5) / C_0

        sh_ks_specular = torch.zeros((self.n_gaussians, (sh_degree + 1) ** 2 - 1, 3), device=device)

        opacities = inv_sigmoid(0.1 * torch.ones((self.n_gaussians, 1), device=device))

        kd_tree = KDTree(point_xyzs.cpu().numpy())
        distances, _ = kd_tree.query(point_xyzs.cpu().numpy(), k=4)
        distances = torch.tensor(distances[:, 1:] ** 2, device=device).mean(axis=-1).sqrt()

        scales = torch.log(torch.clamp_min(distances, 1e-7))[..., None].repeat(1, 3)

        rotations = torch.zeros((self.n_gaussians, 4), device=device)
        rotations[..., 0] = 1

        self.means = torch.nn.Parameter(point_xyzs)
        self.sh_ks_diffuse = torch.nn.Parameter(point_sh_ks_diffuse)
        self.sh_ks_specular = torch.nn.Parameter(sh_ks_specular)
        self.opacities = torch.nn.Parameter(opacities)
        self.scales = torch.nn.Parameter(scales)
        self.rotations = torch.nn.Parameter(rotations)

        self.renderer = GaussianRenderer(sh_degree=self.sh_degree,
                                         sh_update_frequency=sh_update_frequency,
                                         sh_update_start=sh_update_start,
                                         background=[0.0, 0.0, 0.0],
                                         update_statistics=False)

    def get_values(self):
        means = self.means
        sh_ks = torch.cat([self.sh_ks_diffuse[..., None, :], self.sh_ks_specular], dim=1)
        opacities = torch.sigmoid(self.opacities)
        scales = torch.exp(self.scales)
        rotations = torch.nn.functional.normalize(self.rotations)

        return GaussianValues(means, sh_ks, opacities, scales, rotations).contiguous()

    def forward(self, render_info: RenderInfo) -> Tuple[torch.Tensor, dict]:
        render, _, __ = self.renderer(self.get_values, render_info)

        return render.features, {}

    def draw_ui(self):
        with imgui.begin_menu("Background", True) as main_menu:
            if main_menu.opened:
                self.renderer.draw_ui()

