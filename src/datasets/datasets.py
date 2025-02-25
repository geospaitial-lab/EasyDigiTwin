import json
import os
from abc import ABC
from pathlib import Path

import numpy as np
import torch

import geopandas as gpd

from src.datasets.camera_refinements import BaseCameraRefinement
from src.datasets.colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary
from src.datasets.data_loaders import Loader, ConvertedRGBStrategy, BinaryMaskStrategy, DepthStrategy
from src.datasets.data_sampler import SampleMode, DataSampler, InfiniteDataLoader
from src.datasets.cameras import PerspectiveCamera
from src.utils.interfaces import TargetRender
from src.utils.ray_utils import center_poses
from src.utils.rotation_utils import yaw_pitch_roll_to_rot


class Dataset(ABC):
    def __init__(
            self,
            path: Path,
            downsample: float = 1.0,
            sample_mode: SampleMode = SampleMode.ALL,
            latent_embedding_dims: int = None,
            seed: int = None,
            device="cuda",
            return_rays: bool = False,
            batch_size=8192,
            camera_refinement: BaseCameraRefinement = None,
            num_workers: int = 0,
            scale_factor: float = 1.0
    ):
        self.config = {k: v for k, v in vars().items() if k != "self" and k != "__class__"}
        self.config["class"] = type(self).__name__
        self.config["camera_refinement"] = camera_refinement.config if camera_refinement is not None else None
        self.config["path"] = str(path)

        self.path = path
        self.downsample = downsample
        self.sample_mode = sample_mode
        self.latent_embedding_dims = latent_embedding_dims
        self.seed = seed
        self.device = device
        self.return_rays = return_rays
        self.batch_size = batch_size
        self.camera_refinement = camera_refinement
        self.num_workers = num_workers
        self.scale_factor = scale_factor

        self.width = None
        self.height = None
        self.dataloader = None
        self.cameras = None
        self.initial_points_xyz = None
        self.initial_points_rgb = None

    def get_random_batch(self, *args, **kwargs):
        target_dict, image_idx, ray_idx = next(self.dataloader)
        render_infos = []
        target_renders = []
        for i in range(len(image_idx)):
            _ray_idx = ray_idx[i]
            if not torch.is_floating_point(_ray_idx) and torch.numel(_ray_idx) == 1 and _ray_idx == -1:
                _ray_idx = ...
            rgbs = target_dict["rgbs"][i]
            valid_rgbs_mask = target_dict["valid_rgbs_mask"]
            if valid_rgbs_mask == []:
                valid_rgbs_mask = None
            else:
                valid_rgbs_mask = valid_rgbs_mask[i]
            depths = target_dict["depths"]
            if depths == []:
                depths = None
            else:
                depths = depths[i]
            target_render = TargetRender(rgbs=rgbs,
                                         valid_mask=valid_rgbs_mask,
                                         depths=depths)
            render_info = self.cameras(image_idx[i, None], _ray_idx, return_rays=self.return_rays)
            render_info.training = True
            render_infos.append(render_info)
            target_renders.append(target_render)

        return render_infos, target_renders

    def parameters(self):
        return self.cameras.parameters()

    def named_parameters(self):
        return self.cameras.named_parameters()

    def get_callbacks(self):
        return []

    def save(self, path):
        dataset_config_path = os.path.join(path, "dataset_config.json")
        cameras_path = os.path.join(path, "cameras.ckpt")

        with open(dataset_config_path, "w") as file:
            json.dump(self.config, file)
        self.cameras.save(cameras_path)


class ColmapDataset(Dataset):
    def __init__(self,
                 *args,
                 image_base_folder=None,
                 colmap_sub_path=Path("sparse/0"),
                 mask_sub_path=Path("masks"),
                 depth_sub_path=Path("depth"),
                 flip=False,
                 randomization=None,
                 load_mask=False,
                 load_depth=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.config.update({k: v for k, v in vars().items() if k != "self" and k != "__class__"
                            and k != "args" and k != "kwargs"})
        self.config["image_base_folder"] = str(image_base_folder)
        self.config["colmap_sub_path"] = str(colmap_sub_path)
        self.config["mask_sub_path"] = str(mask_sub_path)
        self.config["depth_sub_path"] = str(depth_sub_path)

        self.image_base_folder = image_base_folder
        self.colmap_sub_path = colmap_sub_path
        self.mask_sub_path = mask_sub_path
        self.depth_sub_path = depth_sub_path
        self.load_mask = load_mask
        self.load_depth = load_depth

        f, cx, cy, k1, k2 = self.read_intrinsics()
        poses, data_loader, initial_points_xyz, initial_points_rgb = self.read_meta(flip)

        self.initial_points_xyz = initial_points_xyz
        self.initial_points_rgb = initial_points_rgb

        self.dataloader = data_loader

        self.cameras = PerspectiveCamera(poses, [f, cx, cy, k1, k2], self.width, self.height,
                                         randomization=randomization, latent_embedding_dims=self.latent_embedding_dims,
                                         refinement=self.camera_refinement, device=self.device)

    def read_intrinsics(self):
        camdata = read_cameras_binary(self.path / self.colmap_sub_path / 'cameras.bin')
        self.height = int(camdata[1].height * self.downsample)
        self.width = int(camdata[1].width * self.downsample)

        if camdata[1].model == 'SIMPLE_RADIAL' or camdata[1].model == 'SIMPLE_PINHOLE':
            fx = fy = camdata[1].params[0] * self.downsample
            # cx = camdata[1].params[1] * self.downsample
            # cy = camdata[1].params[2] * self.downsample
            cx = self.width / 2
            cy = self.height / 2
            k1 = 0
            k2 = 0
        elif camdata[1].model == "RADIAL":
            fx = fy = camdata[1].params[0] * self.downsample
            cx = camdata[1].params[1] * self.downsample
            cy = camdata[1].params[2] * self.downsample
            k1 = camdata[1].params[3]
            k2 = camdata[1].params[4]
            # cx = self.width / 2
            # cy = self.height / 2
            # k1 = 0
            # k2 = 0
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0] * self.downsample
            fy = camdata[1].params[1] * self.downsample
            # cx = camdata[1].params[2] * self.downsample
            # cy = camdata[1].params[3] * self.downsample
            cx = self.width / 2
            cy = self.height / 2
            k1 = 0
            k2 = 0
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")

        return (fx + fy) / 2, cx, cy, k1, k2

    def read_meta(self, flip=False):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(self.path / self.colmap_sub_path / 'images.bin')
        img_names = [imdata[k].name for k in imdata]

        self.config["image_names"] = [str(img_name) for img_name in sorted(img_names)]

        perm = np.argsort(img_names)
        if '360_v2' in str(self.path) and self.downsample < 1:  # mipnerf360 data
            if int(1 / self.downsample) in [2, 4, 8]:
                folder = f'images_{int(1 / self.downsample)}'
            else:
                folder = 'images'
        else:
            folder = 'images'
        # read successfully reconstructed images and ignore others

        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3]  # (N_images, 3, 4) cam2world matrices

        colmap_points = read_points3d_binary(self.path / self.colmap_sub_path / 'points3D.bin')
        pts3d = np.array([point.xyz for point in colmap_points.values()])  # (N, 3)
        pts_rgb = np.array([point.rgb for point in colmap_points.values()]) / 255.0

        poses, pts3d = center_poses(poses, pts3d)

        scale = np.linalg.norm(poses[..., 3], axis=-1).max() / self.scale_factor
        poses[..., 3] /= scale
        pts3d /= scale

        if not flip:
            poses[:, 0] *= -1
            poses[:, 2] *= -1
            pts3d[:, 0] *= -1
            pts3d[:, 2] *= -1

        initial_points_xyz = torch.tensor(pts3d, dtype=torch.float)
        initial_points_rgb = torch.tensor(pts_rgb, dtype=torch.float)

        padding = np.tile([0, 0, 0, 1], (len(poses), 1, 1))
        poses = np.concatenate([poses, padding], axis=-2)

        image_base_path = self.path if self.image_base_folder is None else self.image_base_folder

        rgb_paths = [image_base_path / folder / img_name for img_name in sorted(img_names)]
        rgb_strategy = ConvertedRGBStrategy(self.width, self.height, rgb_paths)
        if self.load_mask and (image_base_path / self.mask_sub_path).is_dir():
            mask_paths = [image_base_path / self.mask_sub_path / (img_name + ".png") for img_name in sorted(img_names)]
            mask_strategy = BinaryMaskStrategy(self.width, self.height, mask_paths)
        else:
            mask_strategy = None

        if self.load_depth and (image_base_path / self.depth_sub_path).is_dir():
            depth_paths = [image_base_path / self.depth_sub_path / (img_name + ".npy")
                           for img_name in sorted(img_names)]
            depth_strategy = DepthStrategy(self.width, self.height, depth_paths)

        else:
            depth_strategy = None

        loader = Loader(rgb_strategy, mask_strategy, depth_strategy)

        data_sampler = DataSampler(loader, batch_size=self.batch_size, sample_mode=self.sample_mode, seed=self.seed)

        data_loader = InfiniteDataLoader(data_sampler,
                                         shuffle=True,
                                         pin_memory=True,
                                         num_workers=self.num_workers,
                                         persistent_workers=self.num_workers > 0)

        return poses, data_loader, initial_points_xyz, initial_points_rgb
