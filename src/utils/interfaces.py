import torch
from dataclasses import dataclass, field
from enum import Enum


class ProjectionType(Enum):
    PERSPECTIVE = "perspective"
    ORTHOGRAPHIC = "orthographic"


class SnapMode(Enum):
    SNAP = "snap"
    CENTER_ONLY = "center_only"
    LOOK_AT = "look_at"


@dataclass
class RenderInfo:
    ray_origins_world: torch.Tensor | None = None
    ray_dirs_world: torch.Tensor | None = None
    view_transform: torch.Tensor | None = None
    prespective_transform: torch.Tensor | None = None
    projection_type: ProjectionType = ProjectionType.PERSPECTIVE
    camera_center: torch.Tensor | None = None
    width: int | None = None
    height: int | None = None
    intrinsics: torch.Tensor | None = None
    training: bool = False
    latent_embeddings: torch.Tensor | None = None

    def to(self, device):
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                vars(self)[k] = v.to(device)
        return self

    def retain_grad(self):
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                if v.requires_grad:
                    vars(self)[k].retain_grad()


@dataclass
class TargetRender:
    rgbs: torch.Tensor
    valid_mask: torch.Tensor | None = None
    depths: torch.Tensor | None = None

    def to(self, device):
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                vars(self)[k] = v.to(device)
        return self


class RenderState(Enum):
    RAW = "raw"
    POSTPROCESSED = "postprocessed"
    COMPOSITED = "composited"


class RenderMode(Enum):
    RGB = "rgb"
    DEPTH = "depth"
    OPACITY = "opacity"
    NORMAL = "normal"
    DEPTH_NORMAL = "depth_normal"
    WORLD_POS = "world_pos"
    DISTORTION = "distortion"


@dataclass
class GaussianValues:
    means: torch.Tensor
    sh_ks: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor

    def contiguous(self):
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                vars(self)[k] = v.contiguous()
        return self


@dataclass
class GaussianAuxiliaryRender:
    gaussian_values: GaussianValues | None = None
    distortion_map: torch.Tensor | None = None
    depth_normals: torch.Tensor | None = None
    distances: torch.Tensor | None = None
    view_normals: torch.Tensor | None = None


@dataclass
class Render:
    features: torch.Tensor
    opacities: torch.Tensor
    depths: torch.Tensor
    normals: torch.Tensor | None = None
    world_pos: torch.Tensor | None = None
    auxiliary: GaussianAuxiliaryRender | None = None
    background_auxiliary: dict | None = None
    logs: dict = field(default_factory=dict)
    state: RenderState = RenderState.RAW
