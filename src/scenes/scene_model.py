import json
import os.path
import warnings

import imgui
import torch

from src.scenes.gaussian.gaussian_densifiers import GaussianDensifier
from src.scenes.gaussian.gaussian_models import GaussianModel, load_gaussian_model
from src.scenes.gaussian.gaussian_renderers import GaussianRenderer
from src.utils.interfaces import RenderInfo, Render


class BaseScene:
    def __call__(self, render_info: RenderInfo) -> Render:
        raise NotImplementedError

    def to(self, device):
        raise NotImplementedError

    def get_callbacks(self):
        return []

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, base_path, scene_description, device="cpu"):
        raise NotImplementedError

    def get_roi(self):
        raise NotImplementedError

    def draw_ui(self):
        pass


class EmptyScene(BaseScene):
    def __init__(self):
        self.device = "cpu"

    def to(self, device):
        self.device = device

        return self

    def __call__(self, render_info):
        features = torch.zeros_like(render_info.ray_origins_world, device=self.device)
        opacities = torch.zeros_like(render_info.ray_origins_world[..., 0], device=self.device)
        depths = torch.zeros_like(opacities, device=self.device)

        return Render(features, opacities, depths)

    def save(self, path):
        warnings.warn("Attempting to save EmptyScene! Nothing will happen!")

    def get_roi(self):
        return [-1, -1, -1, 1, 1, 1]


class Gaussian(BaseScene):
    def __init__(self, gaussian_model: GaussianModel, renderer: GaussianRenderer,
                 densifier: GaussianDensifier | None = None):
        self.model = gaussian_model
        self.renderer = renderer
        self.densifier = densifier

    def to(self, device):
        self.model = self.model.to(device)

        return self

    def get_callbacks(self):
        callbacks = self.model.get_callbacks() + self.renderer.get_callbacks()
        if self.densifier is not None:
            callbacks = callbacks + self.densifier.get_callbacks()
        return callbacks

    def get_roi(self):
        return [-1, -1, -1, 1, 1, 1]

    def get_postprocessor(self):
        return self.model.get_postprocessor()

    def __call__(self, render_info: RenderInfo) -> Render:
        render, means_2d, radii = self.renderer(self.model, render_info)
        if render_info.training and self.densifier is not None:
            self.densifier.update_stats(means_2d, radii)
        return render

    def save(self, path):
        scene_description = {"type": "Gaussian",
                             "renderer": self.renderer.__class__.__name__,
                             "model_config": "gaussian_model.ckpt",
                             "model_path": "gaussian_model.ply"}

        with open(os.path.join(path, "scene_model.json"), "w") as file:
            json.dump(scene_description, file)

        self.model.save(path, scene_description["model_config"], scene_description["model_path"])

    @classmethod
    def load(cls, base_path, scene_description, device="cpu"):
        model_path = os.path.join(base_path, scene_description.get("model_path", "gaussian_model.ply"))
        renderer_class = GaussianRenderer

        model = load_gaussian_model(model_path, device=device)
        renderer = renderer_class(sh_degree=model.sh_degree)

        return cls(model, renderer)

    def draw_ui(self):
        with imgui.begin_menu("Scene", True) as main_menu:
            if main_menu.opened:
                self.renderer.draw_ui()


def load_scene_model(path, device="cpu"):
    if os.path.isdir(path):
        base_path = path
        path = os.path.join(path, "scene_model.json")
    else:
        base_path = os.path.dirname(path)

    with open(path, "r") as file:
        scene_description = json.load(file)

    if scene_description["type"] == "Gaussian":
        scene_model = Gaussian.load(base_path, scene_description, device=device)
    else:
        raise NotImplementedError

    return scene_model
