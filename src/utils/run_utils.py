import json
import os

import numpy as np

from src.datasets.cameras import load_cameras, PerspectiveCamera
from src.scenes.backgrounds import ConstantBackgroundModel
from src.scenes.full_scene import FullScene
from src.scenes.gaussian.gaussian_models import load_gaussian_model, GaussianModel
from src.scenes.gaussian.gaussian_renderers import GaussianRenderer
from src.scenes.postprocessors import NoOpPostProcessor
from src.scenes.scene_model import Gaussian, EmptyScene
from src.utils.save_load_utils import load_from_las


def load_scene(scene_path, device="cuda"):
    if not os.path.isdir(scene_path):
        scene_path = os.path.dirname(scene_path)
    scene = FullScene.load(scene_path).to(device)
    scene.background.show_constant_background = True
    train_cams = load_cameras(os.path.join(scene_path, "cameras.ckpt"))

    geo_reference = None
    if os.path.isfile(os.path.join(scene_path, "geo_reference.json")):
        with open(os.path.join(scene_path, "geo_reference.json")) as f:
            geo_reference = json.load(f)
            geo_reference = {"rotation": np.array(geo_reference["rotation"]),
                             "translation": np.array(geo_reference["translation"]),
                             "scaling": np.array(geo_reference["scaling"])}

    return scene, train_cams, geo_reference


def load_pointcloud(path, device="cuda"):
    geo_reference = None
    train_cams = None
    if path.endswith(".las") or path.endswith(".laz"):
        means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular, offset, poses = load_from_las(path)
        model = GaussianModel.from_numpy( means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular)
        geo_reference = {"rotation": np.eye(3),
                         "translation": offset,
                         "scaling": np.ones(1)}

        if poses is not None and len(poses) > 0:
            train_poses = np.array([cam["pose"] for cam in poses])
            train_poses[..., :3, 3] -= offset
            intrinsics = poses[0]["intrinsics"]
            width = poses[0]["width"]
            height = poses[0]["height"]

            train_cams = PerspectiveCamera(train_poses, intrinsics, width, height, device=device)

    else:
        model = load_gaussian_model(path, device=device)
    renderer = GaussianRenderer(sh_degree=model.sh_degree, device=device)
    scene_model = Gaussian(model, renderer)
    scene = FullScene(scene_model, NoOpPostProcessor(), ConstantBackgroundModel([0, 0, 0])).to(device)

    return scene, train_cams, geo_reference


def load_empty_scene(device="cuda"):
    scene = FullScene(EmptyScene(),
                      NoOpPostProcessor(),
                      ConstantBackgroundModel([128, 128, 128])).to(device)

    return scene, None, None
