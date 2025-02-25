import numpy as np
import torch

from src.utils.interfaces import RenderInfo


@torch.amp.autocast('cuda', dtype=torch.float32)
def get_camera_ray_dirs(intrinsics, width, height, device="cpu"):
    f, cx, cy, _, _ = intrinsics
    fx = fy = f
    x, y = torch.meshgrid(torch.arange(width, device=device), torch.arange(height, device=device), indexing="xy")

    x_camera = (x - cx * (width - 1) / width) / fx
    y_camera = (y - cy * (height - 1) / height) / fy

    ray_dirs_camera = torch.stack([x_camera, y_camera, torch.ones_like(x)], dim=-1)
    ray_dirs_camera = ray_dirs_camera / torch.norm(ray_dirs_camera, dim=-1, keepdim=True)
    ray_dirs_camera = ray_dirs_camera.reshape(-1, 3)

    return ray_dirs_camera


@torch.amp.autocast('cuda', dtype=torch.float32)
def get_2d_ray_coordinates(intrinsics, width, height, device="cpu"):
    f, cx, cy, _, _ = intrinsics
    fx = fy = f
    x, y = torch.meshgrid(torch.arange(width, device=device), torch.arange(height, device=device), indexing="xy")

    x_camera = (x - cx * (width - 1) / width) / fx
    y_camera = (y - cy * (height - 1) / height) / fy

    ray_coordinates_2d = torch.stack([x_camera, y_camera], dim=-1)
    ray_coordinates_2d = ray_coordinates_2d.reshape(-1, 2)

    return ray_coordinates_2d


@torch.amp.autocast('cuda', dtype=torch.float32)
def get_rays_from_pose_and_dirs(pose, dirs):
    rotations = pose[..., :3, :3].expand(dirs.shape[0], 3, 3)
    ray_dirs_world = torch.einsum("njk,nk->nj", rotations, dirs)
    ray_origins_world = pose[..., :3, 3].expand_as(ray_dirs_world)

    return ray_origins_world, ray_dirs_world


@torch.amp.autocast('cuda', dtype=torch.float32)
def get_rays(intrinsics, poses, ray_idx, pixel_grid):
    f, cx, cy, k1, k2 = intrinsics

    x, y = pixel_grid

    x_camera = (x.reshape(-1)[ray_idx] - cx + 0.5)
    y_camera = (y.reshape(-1)[ray_idx] - cy + 0.5)
    r2 = (x_camera / f) ** 2 + (y_camera / f) ** 2
    d = 1 / (1 + k1 * r2 + k2 * r2 ** 2)
    x_camera = x_camera * d / f
    y_camera = y_camera * d / f

    assert x_camera.dtype == torch.float32
    assert y_camera.dtype == torch.float32

    ray_dirs_camera = torch.stack([x_camera, y_camera, torch.ones_like(x_camera)], dim=-1)
    ray_dirs_camera = torch.nn.functional.normalize(ray_dirs_camera, p=2, dim=-1)

    assert ray_dirs_camera.dtype == torch.float32

    rotations = poses[..., :3, :3].expand(ray_dirs_camera.shape[0], 3, 3)
    ray_dirs_world = torch.einsum("njk,nk->nj", rotations, ray_dirs_camera)
    ray_origins_world = poses[..., :3, 3].expand_as(ray_dirs_world)

    assert ray_dirs_world.dtype == torch.float32
    assert ray_origins_world.dtype == torch.float32

    return ray_origins_world, ray_dirs_world


@torch.amp.autocast('cuda', dtype=torch.float32)
def get_orthographic_ray_dirs(width, height, device="cpu"):
    ray_dirs_camera = torch.stack([torch.zeros((width * height), device=device),
                                   torch.zeros((width * height), device=device),
                                   torch.ones((width * height), device=device)], dim=-1)
    return ray_dirs_camera


@torch.amp.autocast('cuda', dtype=torch.float32)
def get_orthographic_ray_origins(width, height, view_width, view_height, intrinsics, device="cpu"):
    _, cx, cy, _, _ = intrinsics
    x, y = torch.meshgrid(torch.arange(width, device=device), torch.arange(height, device=device), indexing="xy")

    x_camera = (x - cx * (width - 1) / width) / width * view_width
    y_camera = (y - cy * (height - 1) / height) / height * view_height

    ray_origins_camera = torch.stack([x_camera, y_camera, torch.zeros_like(x)], dim=-1)
    ray_origins_camera = ray_origins_camera.reshape(-1, 3)

    return ray_origins_camera


@torch.amp.autocast('cuda', dtype=torch.float32)
def get_orthographic_view_field(distance, intrinsics, width, height):
    f, _, _, _, _ = intrinsics
    fx = fy = f

    view_x = distance / fx * width
    view_y = distance / fy * height

    return view_x, view_y


@torch.amp.autocast('cuda', dtype=torch.float32)
def get_rays_orthographic(pose, width, height, distance, intrinsics):
    view_x, view_y = get_orthographic_view_field(distance, intrinsics, width, height)
    ray_origins_camera = get_orthographic_ray_origins(width, height, view_x, view_y, intrinsics, device=pose.device)
    ray_dirs_camera = get_orthographic_ray_dirs(width, height, device=pose.device)
    ray_origins_world = torch.einsum("jk,nk->nj", pose[:3, :3], ray_origins_camera)
    ray_origins_world = ray_origins_world + pose[:3, 3]
    ray_dirs_world = torch.einsum("jk,nk->nj", pose[:3, :3], ray_dirs_camera)

    return ray_origins_world, ray_dirs_world


@torch.amp.autocast('cuda', dtype=torch.float32)
def get_view_transform(pose):
    view_mat = torch.eye(4)
    view_mat[:3, :3] = pose[:3, :3].T
    view_mat[:3, 3] = pose[:3, :3].T @ -pose[:3, 3]

    return view_mat.to(torch.float32)


@torch.amp.autocast('cuda', dtype=torch.float32)
def get_perspective_transform(f, cx, cy, width, height, near, far):
    zero = torch.tensor(0.0, device=f.device)
    projection_mat = torch.stack([torch.stack([(2 * f) / (width-1), zero,
                                               ((width-1) - 2 * cx * (width - 1) / width) / (width - 1), zero]),
                                  torch.stack([zero, (2 * f) / (height-1),
                                               ((height-1) - 2 * cy * (height - 1) / height) / (height - 1), zero]),
                                  torch.tensor([0, 0, -(far + near) / (far - near), -2 * near * far / (far - near)],
                                               device=f.device),
                                  torch.tensor([0, 0, -1, 0], device=f.device)])

    return projection_mat.to(torch.float32)


@torch.amp.autocast('cuda', dtype=torch.float32)
def get_orthographic_transform(f, cx, cy, width, height, near, far, distance):
    projection_mat = torch.tensor([[(2 * f) / (distance * (width - 1)), 0, 0,
                                    -((width - 1) - 2 * cx * (width - 1) / width) / (width - 1)],
                                   [0, (2 * f) / (distance * (height - 1)), 0,
                                    -((height-1) - 2 * cy * (height - 1) / height) / (height - 1)],
                                   [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                                   [0, 0, 0, 1]])

    return projection_mat.to(torch.float32)


@torch.amp.autocast('cuda', dtype=torch.float32)
def get_intrinsics_matrix(f, cx, cy):
    zero = torch.tensor(0.0, device=f.device)
    intrinsics_mat = torch.stack([torch.stack([f, zero, cx], dim=-1),
                                  torch.stack([zero, f, cy], dim=-1),
                                  torch.tensor([0, 0, 1], device=f.device, dtype=f.dtype)])

    return intrinsics_mat


@torch.amp.autocast('cuda', dtype=torch.float32)
def info_to_rays(render_info: RenderInfo, normalize=True, device="cpu"):
    pixel_grid = torch.meshgrid(torch.arange(int(render_info.width), device=device),
                                torch.arange(int(render_info.height), device=device), indexing="xy")

    f, cx, cy, k1, k2 = render_info.intrinsics

    x, y = pixel_grid

    x_camera = (x.reshape(-1) - cx * (render_info.width - 1) / render_info.width)
    y_camera = (y.reshape(-1) - cy * (render_info.height - 1) / render_info.height)
    r2 = (x_camera / f) ** 2 + (y_camera / f) ** 2
    d = 1 / (1 + k1 * r2 + k2 * r2 ** 2)
    x_camera = x_camera * d / f
    y_camera = y_camera * d / f

    ray_dirs_camera = torch.stack([x_camera, y_camera, torch.ones_like(x_camera)], dim=-1)
    if normalize:
        ray_dirs_camera = torch.nn.functional.normalize(ray_dirs_camera, p=2, dim=-1)

    pose = render_info.view_transform.inverse()
    rotations = pose[:3, :3]
    ray_dirs_world = torch.einsum("jk,nk->nj", rotations, ray_dirs_camera)
    ray_origins_world = pose[:3, 3].expand_as(ray_dirs_world)

    return ray_origins_world, ray_dirs_world


@torch.amp.autocast('cuda', dtype=torch.float32)
def depth_to_points(depths, render_info: RenderInfo, z_depth=False):
    ray_origins_world, ray_dirs_world = info_to_rays(render_info, normalize=not z_depth, device=depths.device)
    points = ray_origins_world + ray_dirs_world * depths[..., None]

    return points


@torch.amp.autocast('cuda', dtype=torch.float32)
def depth_to_normals(depths, render_info: RenderInfo, z_depth=False):
    points = depth_to_points(depths, render_info, z_depth)
    points = points.reshape(int(render_info.height), int(render_info.width), 3)

    normals = torch.zeros_like(points)

    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    normals[1:-1, 1:-1] = torch.nn.functional.normalize(torch.cross(dy, dx, dim=-1), dim=-1)

    return normals


@torch.amp.autocast('cuda', dtype=torch.float32)
def project_points(points, render_info: RenderInfo):
    perspective_transform = render_info.prespective_transform.clone()
    perspective_transform[..., 2] *= -1
    points_view = torch.einsum("jk,nk->nj",
                               render_info.view_transform,
                               torch.cat([points, torch.ones_like(points[..., :1])], dim=-1))
    points_projected = torch.einsum("jk,nk->nj",
                                    perspective_transform,
                                    points_view)

    points_projected = points_projected[..., :3] / (points_projected[..., 3:] + 1e-8)

    return points_view, points_projected


@torch.amp.autocast('cuda', dtype=torch.float32)
def reproject_points_from_view(points, depths, render_info: RenderInfo, z_depth=False):
    points_view, points_projected = project_points(points, render_info)

    valid_mask = ((points_projected[..., 0] >= -1)
                  & (points_projected[..., 0] <= 1)
                  & (points_projected[..., 1] >= -1)
                  & (points_projected[..., 1] <= 1)
                  & (points_projected[..., 2] >= -1))

    depth_image = depths.reshape(-1, 1, int(render_info.height), int(render_info.width))

    sampled_depths = torch.nn.functional.grid_sample(depth_image,
                                                     points_projected[..., :2].reshape(1, -1, 1, 2),
                                                     mode="bilinear",
                                                     padding_mode="border").reshape(-1)
    if z_depth:
        reprojected_points_view = points_view[..., :3] / (points_view[..., 2:3] + 1e-8) * sampled_depths[..., None]
    else:
        reprojected_points_view = torch.nn.functional.normalize(points_view[..., :3],
                                                                p=2, dim=-1) * sampled_depths[..., None]

    reprojected_points = torch.einsum("jk,nk->nj",
                               render_info.view_transform.inverse(),
                               torch.cat([reprojected_points_view,
                                          torch.ones_like(reprojected_points_view[..., :1])], dim=-1))[..., :3]

    return reprojected_points, valid_mask

def normalize_np(v):
    # From https://github.com/kwea123/ngp_pl/blob/1b49af1856a276b236e0f17539814134ed329860/datasets/ray_utils.py#L103
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses, pts3d=None):
    # From https://github.com/kwea123/ngp_pl/blob/1b49af1856a276b236e0f17539814134ed329860/datasets/ray_utils.py#L108
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of 3d point cloud (if None, center of cameras).
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    if pts3d is not None:
        center = pts3d.mean(0)
    else:
        center = poses[..., 3].mean(0)

    # 2. Compute the z axis
    z = normalize_np(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize_np(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, pts3d=None):
    # From https://github.com/kwea123/ngp_pl/blob/1b49af1856a276b236e0f17539814134ed329860/datasets/ray_utils.py#L150
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """

    pose_avg = average_poses(poses, None)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    if pts3d is not None:
        pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T
        return poses_centered, pts3d_centered[..., :3]

    return poses_centered
