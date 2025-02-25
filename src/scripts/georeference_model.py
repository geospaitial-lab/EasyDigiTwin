import json
import os
import warnings

import numpy as np
import pandas as pd
import roma
import torch

from src.datasets.cameras import load_cameras
from src.scenes.scene_model import load_scene_model
from src.utils.rotation_utils import georeference_poses, georeference_gaussians
from src.utils.save_load_utils import save_to_las

warnings.filterwarnings('ignore', message="TypedStorage is deprecated")

scene_path = "/path/to/saved/scene"
reference_file_path = "/path/to/geo_reference.txt"

if __name__ == "__main__":
    model_path = os.path.join(scene_path, "scene_model.json")
    save_path = os.path.join(scene_path, "referenced_gaussians.laz")

    reference_df = pd.read_csv(reference_file_path, sep=" ", header=None, names=["filename", "x", "y", "z"])

    with torch.no_grad():
        cameras = load_cameras(os.path.join(scene_path, "cameras.ckpt"))
        poses = cameras.poses(...).cpu().to(torch.float64)
        intrinsics = [float(i) for i in cameras.intrinsics(torch.tensor([0]))]
        width = int(cameras.width)
        height = int(cameras.height)
        model_cam_centers = poses[:, :3, 3]

    dataset_config_path = os.path.join(scene_path, "dataset_config.json")
    with open(dataset_config_path, "r") as file:
        dataset_config = json.load(file)

    image_names = dataset_config["image_names"]

    reference_cam_centers = torch.tensor(np.array([reference_df[reference_df["filename"] == name].to_numpy()[0][1:4].astype(float)
                                          for name in image_names]), dtype=torch.float64)

    with torch.no_grad():

        R, t, s = roma.rigid_points_registration(model_cam_centers, reference_cam_centers, compute_scaling=True)

        R = R.numpy()
        t = t.numpy()
        s = s.numpy()

    geo_reference = {"rotation": R.tolist(), "translation": t.tolist(), "scaling": s.tolist()}

    with open(os.path.join(scene_path, "geo_reference.json"), "w") as file:
        json.dump(geo_reference, file)

    cam_poses = poses.cpu().numpy()
    ref_poses = georeference_poses(cam_poses, R, t, s)

    scene_model = load_scene_model(model_path, device="cpu")
    means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular = scene_model.model.to_numpy()
    means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular = georeference_gaussians(
        means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular,
        R, t, s
    )

    pose_dict_list = [{
        "file": image_names[i],
        "pose": ref_poses[i].tolist(),
        "width": width,
        "height": height,
        "intrinsics": intrinsics
    } for i in range(len(image_names))]

    save_to_las(save_path, means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular,
                epsg="25832", poses=pose_dict_list)
