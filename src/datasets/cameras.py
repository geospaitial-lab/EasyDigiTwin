import json

import numpy as np
import torch

from src.datasets.camera_refinements import BaseCameraRefinement, randomize_poses, load_refinement
from src.utils.interfaces import RenderInfo
from src.utils.ray_utils import get_rays, get_view_transform, get_perspective_transform
from src.utils.rotation_utils import yaw_pitch_roll_to_rot
from src.utils.save_load_utils import save_to_checkpoint


class TrainableCameras(torch.nn.Module):
    def __init__(self, initial_poses, initial_intrinsics, width, height,
                 cams_to_use=None, scale_factor=1.0, device="cuda", randomization=None, latent_embedding_dims=None,
                 near=0.002, far=10.0, refinement: BaseCameraRefinement | None = None):
        super(TrainableCameras, self).__init__()
        if cams_to_use is None:
            cams_to_use = ["main"]
        self.config = {"class": type(self).__name__,
                       "n_positions": len(initial_poses),
                       "n_cams": len(cams_to_use),
                       "cams_to_use": cams_to_use,
                       "width": width,
                       "height": height,
                       "near": near,
                       "far": far,
                       "scale_factor": scale_factor,
                       "latent_embedding_dims": latent_embedding_dims,
                       "refinement": refinement.config if refinement is not None else None}

        self.near = near
        self.far = far
        self.latent_embedding_dims = latent_embedding_dims
        self.downsample = 1.0

        self.register_buffer("orig_frame_poses", torch.tensor(initial_poses, device=device, dtype=torch.float))
        self.register_buffer("orig_intrinsics", torch.tensor(initial_intrinsics, device=device, dtype=torch.float))
        self.register_buffer("orig_camera_alignments",
                             torch.tensor(np.array([self.get_camera_to_frame_transform(cam_name)
                                                    for cam_name in cams_to_use]),
                                          device=device, dtype=torch.float))
        self.register_buffer("width", torch.tensor(width, device=device))
        self.register_buffer("height", torch.tensor(height, device=device))
        self.register_buffer("n_cams", torch.tensor(len(cams_to_use), device=device))
        self.register_buffer("n_positions", torch.tensor(len(initial_poses), device=device))
        self.register_buffer("scale_factor", torch.tensor(scale_factor, device=device))

        self.refinement = refinement
        if self.refinement is not None:
            self.refinement.setup_parameters(self.n_positions, self.n_cams, device)

        self.register_buffer("gt_poses", self.poses(...))
        if randomization is not None:
            self.orig_frame_poses = randomize_poses(torch.tensor(initial_poses,
                                                                 device=device,
                                                                 dtype=torch.float),
                                                    random_rot=randomization[0],
                                                    random_trans=randomization[1])
        self.register_buffer("orig_poses", self.poses(...))

        if self.latent_embedding_dims is not None:
            self.latent_embedding = torch.nn.Embedding(num_embeddings=len(self.orig_poses),
                                                       embedding_dim=self.latent_embedding_dims)
        else:
            self.latent_embedding = lambda x: None

    def forward(self, image_idx, ray_idx, return_rays=True) -> RenderInfo:
        image_idx = torch.tensor(image_idx)

        poses, intrinsics = self.poses_and_intrinsics(image_idx)

        ray_origins_world = None
        ray_dirs_world = None
        if return_rays:
            if ray_idx is ...:
                ray_idx = torch.arange((self.width * self.downsample).to(int) * (self.height * self.downsample).to(int))
            else:
                ray_idx = torch.tensor(ray_idx)
            pixel_grid = torch.meshgrid(torch.arange((self.width * self.downsample).to(int),
                                                     device=self.width.device),
                                        torch.arange((self.height * self.downsample).to(int),
                                                     device=self.height.device),
                                        indexing="xy")
            ray_origins_world, ray_dirs_world = get_rays(intrinsics, poses, ray_idx, pixel_grid)

        latent_embeddings = None
        if self.latent_embedding_dims is not None:
            latent_embeddings = self.latent_embedding(image_idx.expand_as(ray_dirs_world[..., 0]))

        view_transform = self.view_transform(poses[0])
        perspective_transform = self.perspective_transform(intrinsics)
        camera_center = poses[0, :3, 3]

        render_info = RenderInfo(ray_origins_world=ray_origins_world, ray_dirs_world=ray_dirs_world,
                                 latent_embeddings=latent_embeddings, intrinsics=intrinsics,
                                 width=(self.width * self.downsample).to(int),
                                 height=(self.height * self.downsample).to(int),
                                 view_transform=view_transform, prespective_transform=perspective_transform,
                                 camera_center=camera_center)

        return render_info

    def intrinsics(self, image_idx):
        return self.poses_and_intrinsics(image_idx)[1]

    def poses(self, image_idx):
        if image_idx is ...:
            image_idx = torch.arange(self.n_positions * self.n_cams, device=self.orig_frame_poses.device)
        return self.poses_and_intrinsics(image_idx)[0]

    def frame_poses(self):
        image_idx = torch.arange(self.n_positions, device=self.orig_frame_poses.device) * self.n_cams
        frame_idx = image_idx // self.n_cams
        cam_idx = image_idx % self.n_cams
        f2w = self.orig_frame_poses[frame_idx]
        c2f = self.orig_camera_alignments[cam_idx]
        intrinsics = self.orig_intrinsics.repeat(len(cam_idx), 1).T

        if self.refinement is not None:
            f2w, c2f, intrinsics = self.refinement(f2w, c2f, intrinsics, frame_idx)

        return f2w

    @torch.amp.autocast('cuda', dtype=torch.float32)
    def poses_and_intrinsics(self, image_idx):
        image_idx = image_idx.to(self.orig_frame_poses.device)
        frame_idx = image_idx // self.n_cams
        cam_idx = image_idx % self.n_cams

        f2w = self.orig_frame_poses[frame_idx]
        c2f = self.orig_camera_alignments[cam_idx]
        intrinsics = self.orig_intrinsics

        if self.refinement is not None:
            f2w, c2f, intrinsics = self.refinement(f2w, c2f, intrinsics, image_idx)
        if self.n_cams > 1:
            poses = f2w @ c2f
        else:
            poses = f2w
        poses[..., :3, 3] /= self.scale_factor  # scale here to include alignment in scaling

        f, cx, cy, k1, k2 = intrinsics
        f = f * self.downsample
        cx = cx * self.downsample
        cy = cy * self.downsample
        if f.dim() > 0:  # sadly very hacky
            f = f[0]
        if cx.dim() > 0:
            cx = cx[0]
        if cy.dim() > 0:
            cy = cy[0]
        if k1.dim() > 0:
            k1 = k1[0]
        if k2.dim() > 0:
            k2 = k2[0]
        intrinsics = f, cx, cy, k1, k2

        return poses, intrinsics

    @torch.amp.autocast('cuda', dtype=torch.float32)
    def view_transform(self, pose):
        view_mat = get_view_transform(pose)

        return view_mat

    @torch.amp.autocast('cuda', dtype=torch.float32)
    def perspective_transform(self, intrinsics):
        f, cx, cy, _, _ = intrinsics
        projection_mat = get_perspective_transform(f, cx, cy,
                                                   (self.width * self.downsample).to(int),
                                                   (self.height * self.downsample).to(int),
                                                   self.near, self.far)

        return projection_mat

    def __len__(self):
        return len(self.orig_poses)

    def get_camera_to_frame_transform(self, cam_name):
        raise NotImplementedError

    def save(self, path):
        save_to_checkpoint(self, path)


def load_cameras(path, cameras=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    config = checkpoint["config"]
    camera_class = eval(config["class"])
    dummy_poses = np.zeros((config["n_positions"], 4, 4))
    dummy_intrinsics = np.zeros(5)

    refinement = None
    if "refinement" in config and config["refinement"] is not None:
        refinement = load_refinement(config["refinement"])

    if cameras is None:
        cameras = camera_class(dummy_poses, dummy_intrinsics,
                               config["width"], config["height"],
                               config["cams_to_use"],
                               scale_factor=config["scale_factor"],
                               latent_embedding_dims=config["latent_embedding_dims"],
                               near=config["near"],
                               far=config["far"],
                               refinement=refinement)
    cameras.load_state_dict(checkpoint["state_dict"], strict=False)

    return cameras


class PerspectiveCamera(TrainableCameras):
    def get_camera_to_frame_transform(self, cam_name):
        return np.eye(4)
