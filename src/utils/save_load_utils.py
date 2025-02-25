import datetime
import json
import math
import re
import warnings
from typing import Tuple

import laspy
import numpy as np
import pandas as pd
import pyproj
import roma
import torch
from pyntcloud import PyntCloud

from src.utils.general_utils import C_0, np_sigmoid


def save_to_checkpoint(module: torch.nn.Module, save_path):
    save_dict = {"config": module.config,
                 "state_dict": module.state_dict()}

    torch.save(save_dict, save_path)


def load_from_checkpoint(path, get_class_func, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    module_class = checkpoint["config"].pop("class")
    module_class = get_class_func(module_class)
    module = module_class(**checkpoint["config"])
    module.to(device)
    module.load_state_dict(checkpoint["state_dict"])

    return module

def save_to_ply(ply_path,
                means: np.ndarray,
                opacities: np.ndarray,
                rotations: np.ndarray,
                scales: np.ndarray,
                sh_ks_diffuse: np.ndarray,
                sh_ks_specular: np.ndarray,
                rgb_format: str = "rgb"):

    if rgb_format == "sh":
        r_key = "f_dc_0"
        g_key = "f_dc_1"
        b_key = "f_dc_2"
    else:
        sh_ks_diffuse = sh_ks_diffuse * C_0 + 0.5

        r_key = "r"
        g_key = "g"
        b_key = "b"

    pcd_df = pd.DataFrame({"x": means[:, 0], "y": means[:, 1], "z": means[:, 2]})

    if rgb_format == "sh":
        pcd_df[["nx", "ny", "nz"]] = np.zeros_like(means)

    pcd_df[[r_key, g_key, b_key]] = sh_ks_diffuse

    if sh_ks_specular.shape[-2] > 0:
        n_sh_ks = sh_ks_specular.shape[-2] * 3

        if rgb_format == "sh":
            sh_ks_specular = np.moveaxis(sh_ks_specular, -1, -2)

        sh_ks_specular = sh_ks_specular.reshape(-1, n_sh_ks)

        pcd_df[[f"f_rest_{i}" for i in range(n_sh_ks)]] = sh_ks_specular

    pcd_df["opacity"] = opacities
    pcd_df[["scale_0", "scale_1", "scale_2"]] = scales
    pcd_df[["rot_0", "rot_1", "rot_2", "rot_3"]] = rotations

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        pcd = PyntCloud(pcd_df)
        pcd.to_file(ply_path)


def load_from_ply(ply_path):
    pcd = PyntCloud.from_file(ply_path)
    pcd_df = pcd.points

    regex = re.compile("f_rest_(\d+)")
    max_sh = max([int(re.search(regex, col).group(1)) for col in pcd_df.columns if regex.match(col)]) + 1
    sh_degree = int(math.sqrt(max_sh//3 + 1) - 1)

    means = np.stack([pcd_df["x"], pcd_df["y"], pcd_df["z"]], axis=-1)

    if "f_dc_0" in pcd_df.columns:
        rgb_format = "sh"
    elif "r" in pcd_df.columns:
        rgb_format = "rgb"
    else:
        raise ValueError("Invalid RGB format")

    if rgb_format == "sh":
        r_key = "f_dc_0"
        g_key = "f_dc_1"
        b_key = "f_dc_2"
    else:
        r_key = "r"
        g_key = "g"
        b_key = "b"
    sh_ks_diffuse = np.stack([pcd_df[r_key], pcd_df[g_key], pcd_df[b_key]], axis=-1)
    if rgb_format != "sh":
        sh_ks_diffuse = (sh_ks_diffuse - 0.5) / C_0
    if sh_degree > 0:
        sh_ks_specular = np.stack([pcd_df[f"f_rest_{i}"] for i in range(((sh_degree + 1) ** 2 - 1) * 3)],
                                  axis=-1).reshape(-1, (sh_degree + 1) ** 2 - 1, 3)
    else:
        sh_ks_specular = np.empty([sh_ks_diffuse.shape[0], 0, 3])

    if rgb_format == "sh":
        sh_ks_specular = np.moveaxis(sh_ks_specular.reshape(-1, 3, (sh_degree + 1) ** 2 - 1), -1, -2)

    opacities = pcd_df["opacity"].to_numpy()[..., None]
    scales = np.stack([pcd_df["scale_0"], pcd_df["scale_1"], pcd_df["scale_2"]], axis=-1)
    rotations = np.stack([pcd_df["rot_0"], pcd_df["rot_1"], pcd_df["rot_2"], pcd_df["rot_3"]], axis=-1)

    return means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular


class GaussianSplattingVLR(laspy.vlrs.BaseKnownVLR):
    def __init__(self, sh_degree):
        super().__init__()
        self.sh_degree = sh_degree
        self.georeferenced = False
        self.poses = None

    @staticmethod
    def official_user_id() -> str:
        return "geospaitial-lab"

    @staticmethod
    def official_record_ids() -> Tuple[int, ...]:
        return (0, )

    def record_data_bytes(self) -> bytes:
        return json.dumps(dict(self)).encode("ascii")

    def parse_record_data(self, record_data: bytes) -> None:
        attr_dict = json.loads(record_data.decode("ascii"))

        for k, v in attr_dict.items():
            setattr(self, k, v)

    def __repr__(self):
        return "<GaussianSplattingVLR>"

    def __iter__(self):
        for var in vars(self).items():
            if not var[0].startswith("_"):
               yield var


def save_to_las(las_path,
                means: np.ndarray,
                opacities: np.ndarray,
                rotations: np.ndarray,
                scales: np.ndarray,
                sh_ks_diffuse: np.ndarray,
                sh_ks_specular: np.ndarray,
                epsg: str | None = None,
                poses: list[dict] | None = None):
    sh_degree = int(math.sqrt(sh_ks_specular.shape[-2] + 1) - 1)

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.offsets = np.min(means, axis=0)
    header.scales = [0.001, 0.001, 0.001]
    if epsg is not None:
        header.add_crs(pyproj.CRS.from_epsg(epsg))
    header.system_identifier = "M0000"
    header.generating_software = "EasyDigiTwin V-0.0.1"
    header.creation_date = datetime.date.today()

    las = laspy.LasData(header)
    gaussian_splatting_vlr = GaussianSplattingVLR(sh_degree)
    if poses is not None:
        gaussian_splatting_vlr.poses = poses
    if epsg is not None:
        gaussian_splatting_vlr.georeferenced = True

    las.evlrs = laspy.vlrs.vlrlist.VLRList()
    las.evlrs.append(gaussian_splatting_vlr)

    las.synthetic = np.ones(len(means), dtype=bool)
    las.withheld =  np_sigmoid(opacities[..., 0]) < 0.15

    las.x = means[:, 0]
    las.y = means[:, 1]
    las.z = means[:, 2]

    r = (sh_ks_diffuse[..., 0] * C_0 + 0.5)
    g = (sh_ks_diffuse[..., 1] * C_0 + 0.5)
    b = (sh_ks_diffuse[..., 2] * C_0 + 0.5)
    r = np.clip(r, 0, 1) ** 1 / 1.2
    g = np.clip(g, 0, 1) ** 1 / 1.2
    b = np.clip(b, 0, 1) ** 1 / 1.2

    las.red = r * 2 ** 16
    las.green = g * 2 ** 16
    las.blue = b * 2 ** 16

    las.add_extra_dims([laspy.ExtraBytesParams(name="opacity",
                                               type="f4",
                                               description="raw opacities"),
                        laspy.ExtraBytesParams(name="scale",
                                               type="3f4",
                                               description="raw scales"),
                        laspy.ExtraBytesParams(name="rotation_x",
                                               type="f4",
                                               description="rotation quat x"),
                        laspy.ExtraBytesParams(name="rotation_y",
                                               type="f4",
                                               description="rotation quat y"),
                        laspy.ExtraBytesParams(name="rotation_z",
                                               type="f4",
                                               description="rotation quat z"),
                        laspy.ExtraBytesParams(name="rotation_w",
                                               type="f4",
                                               description="rotation quat w"),
                        ])

    las.opacity = opacities[..., 0]
    las.scale = scales
    las.rotation_w = rotations[:, 0]
    las.rotation_x = rotations[:, 1]
    las.rotation_y = rotations[:, 2]
    las.rotation_z = rotations[:, 3]

    for deg in range((sh_degree + 1) ** 2):
        las.add_extra_dim(laspy.ExtraBytesParams(name=f"sh_{deg}",
                                                 type="3f4",
                                                 description=f"sh coeffs no. {deg}"))

        if deg == 0:
            las[f"sh_{deg}"] = sh_ks_diffuse
        else:
            las[f"sh_{deg}"] = sh_ks_specular[..., deg - 1, :]

    las.write(las_path)


def load_from_las(las_path):
    las = laspy.read(las_path)

    gaussian_splatting_vlr = GaussianSplattingVLR(3)
    for vlr in las.evlrs:
        if vlr.user_id == "geospaitial-lab" and vlr.record_id == 0:
            gaussian_splatting_vlr.parse_record_data(vlr.record_data)

    sh_degree = gaussian_splatting_vlr.sh_degree

    poses = gaussian_splatting_vlr.poses

    means = las.xyz
    opacities = las.opacity[..., None]

    rotations = [las.rotation_w, las.rotation_x, las.rotation_y, las.rotation_z]
    rotations = np.stack(rotations, axis=-1)

    scales = las.scale
    sh_ks_diffuse = las["sh_0"]

    sh_ks_specular = []

    for deg in range((sh_degree + 1) ** 2 - 1):
        sh_ks_specular.append(las[f"sh_{deg + 1}"])

    sh_ks_specular = np.stack(sh_ks_specular, axis=-2)

    scene_offset = np.array([0.0, 0.0, 0.0])
    if gaussian_splatting_vlr.georeferenced:
        scene_offset = las.header.offsets
        means = means - scene_offset

    return (means.copy(), opacities.copy(), rotations.copy(), scales.copy(), sh_ks_diffuse.copy(),
            sh_ks_specular.copy(), scene_offset, poses)


def save_pointcloud(save_path, means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular, rgb_format="sh"):
    if save_path.endswith(".ply"):
        save_to_ply(save_path, means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular,
                    rgb_format=rgb_format)
    elif save_path.endswith(".las") or save_path.endswith(".laz"):
        save_to_las(save_path, means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular)
