import math

import numpy as np
import torch
import roma
from e3nn import o3


def skew_matrix(vector):
    b_zero = torch.zeros(vector.shape[:-1], device=vector.device)  # (batchsize,)

    skew_row_0 = torch.stack([b_zero, -vector[..., 2], vector[..., 1]], dim=-1)  # (batchsize, 3)
    skew_row_1 = torch.stack([vector[..., 2], b_zero, -vector[..., 0]], dim=-1)  # (batchsize, 3)
    skew_row_2 = torch.stack([-vector[..., 1], vector[..., 0], b_zero], dim=-1)  # (batchsize, 3)

    return torch.stack([skew_row_0, skew_row_1, skew_row_2], dim=-2)  # (batchsize, 3, 3)


@torch.amp.autocast('cuda', dtype=torch.float32)
def axisangle_to_rot(axisangle):
    """
    Convert axisangle vectors to rotation matrices.
    Code inspired by https://github.com/kwea123/ngp_pl/blob/2bf4b6a2aeaf5e481fc5334f4f309585d937574d/datasets/ray_utils.py#L74
    Explanation: https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    :param axisangle: axisangle vector. Direction specifies plane to rotate in, length specifies angle.
                      (batchsize, 3) or (3,)
    :return: Rotation Matrix for axisangles. (batchsize, 3, 3) or (3, 3)
    """

    if axisangle.dim() == 1:
        batchsize = 1
        unbatch = True
    else:
        batchsize = axisangle.size(dim=0)
        unbatch = False

    axisangle = torch.reshape(axisangle, (batchsize, -1))  # (batchsize, 3)

    angle = torch.norm(axisangle, dim=-1)  # (batchsize, 1)

    axis = axisangle / torch.clamp(angle, min=1e-7)[:, None]  # (batchsize, 3)

    skew_axis = skew_matrix(axis)

    rot_mat = (torch.eye(3, device=axisangle.device) +
               torch.sin(angle)[:, None, None] * skew_axis +
               (1 - torch.cos(angle))[:, None, None] * (skew_axis @ skew_axis))  # (batchsize, 3, 3)

    if unbatch:
        rot_mat = torch.reshape(rot_mat, (3, 3))

    return rot_mat


def yaw_pitch_roll_to_rot(yaw, pitch, roll, degrees=True):
    yaw = np.deg2rad(yaw) if degrees else yaw
    pitch = np.deg2rad(pitch) if degrees else pitch
    roll = np.deg2rad(roll) if degrees else roll

    r_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    r_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    r_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

    return r_yaw @ r_pitch @ r_roll


@torch.amp.autocast('cuda', dtype=torch.float32)
def rot_mat_from_6d(rot_6d):
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]

    b1 = torch.nn.functional.normalize(a1, p=2, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, p=2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    rot_mat = torch.concatenate([b1[..., None], b2[..., None], b3[..., None]], dim=-1)

    return rot_mat


@torch.amp.autocast('cuda', dtype=torch.float32)
def rotation_translation_to_screw_axis(rotation, translation):
    """
    adapted from https://github.com/jonbarron/camp_zipnerf/blob/16206bd88f37d5c727976557abfbd9b4fa28bbe1/internal/rigid_body.py#L193

    """
    eps = torch.finfo(torch.float32).eps
    w = roma.rotmat_to_rotvec(rotation)
    theta_sq = torch.sum(w * w, dim=-1)
    theta = torch.sqrt(torch.clip(theta_sq, min=eps))

    skew = skew_matrix(w / theta[..., None])

    g_inv = (torch.eye(3, device=theta.device, dtype=theta.dtype) +
             theta[..., None, None] * -skew / 2.0 +
             (1.0 - 0.5 * theta / torch.tan(theta / 2.0))[..., None, None] * skew @ skew)

    v = (g_inv @ translation[..., None]).squeeze(-1)
    v = torch.where(theta_sq[..., None] > eps, v, translation)  # for numerical stability. theta = 0 means pure translation

    skrew_axis = torch.cat([w, v], dim=-1)

    return skrew_axis


def skrew_axis_to_rotation_translation(skrew_axis):
    """
    adapted from https://github.com/jonbarron/camp_zipnerf/blob/16206bd88f37d5c727976557abfbd9b4fa28bbe1/internal/rigid_body.py#L157
    """
    eps = torch.finfo(torch.float32).eps
    w = skrew_axis[..., :3]
    v = skrew_axis[..., 3:]

    rotation = roma.rotvec_to_rotmat(w)

    theta_sq = torch.sum(w * w, dim=-1)
    theta = torch.sqrt(torch.clip(theta_sq, min=eps))

    skew = skew_matrix(w / theta[..., None])

    g = (theta[..., None, None] * torch.eye(3, device=theta.device, dtype=theta.dtype)[None, ...] +
         (1.0 - torch.cos(theta))[..., None, None] * skew +
         (theta - torch.sin(theta))[..., None, None] * skew @ skew) / theta[..., None, None]

    translation = (g @ v[..., None]).squeeze(-1)
    translation = torch.where(theta_sq[..., None] > (eps ** 2), translation, v)  # for numerical stability

    return rotation, translation


def quaternion_to_rotmat(q):
    x, y, z, w = q.unbind(dim=-1)

    yy = y * y
    zz = z * z
    ww = w * w
    yz = y * z
    wx = w * x
    wy = w * y
    zx = z * x
    zw = z * w
    yx = y * x

    rot_mat = torch.stack([
        torch.stack([1 - 2 * (zz + ww), 2 * (yz - wx), 2 * (wy + zx)], dim=-1),
        torch.stack([2 * (yz + wx), 1 - 2 * (ww + yy), 2 * (zw - yx)], dim=-1),
        torch.stack([2 * (wy - zx), 2 * (zw + yx), 1 - 2 * (yy + zz)], dim=-1)
    ], dim=-2)

    return rot_mat


def rotate_sh_ks(sh_ks_specular, rot_mat):
    rot_angles = roma.rotmat_to_euler("ZYZ", torch.from_numpy(rot_mat))
    sh_degree = int(math.sqrt(sh_ks_specular.shape[-2] + 1) - 1)
    einsum_notation = "i j, n k j -> n k i"
    rotated_sh_ks = []

    if sh_degree > 0:
        wigner_d_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).cpu().numpy()

        sh_ks_deg_1 = np.moveaxis(sh_ks_specular[:, 0:3, :], -1, -2)
        sh_ks_deg_1 = np.moveaxis(np.einsum(einsum_notation, wigner_d_1, sh_ks_deg_1), -1, -2)

        rotated_sh_ks.append(sh_ks_deg_1)

    if sh_degree > 1:
        wigner_d_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).cpu().numpy()

        sh_ks_deg_2 = np.moveaxis(sh_ks_specular[:, 3:8, :], -1, -2)
        sh_ks_deg_2 = np.moveaxis(np.einsum(einsum_notation, wigner_d_2, sh_ks_deg_2), -1, -2)

        rotated_sh_ks.append(sh_ks_deg_2)

    if sh_degree > 2:
        wigner_d_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).cpu().numpy()

        sh_ks_deg_3 = np.moveaxis(sh_ks_specular[:, 8:15, :], -1, -2)
        sh_ks_deg_3 = np.moveaxis(np.einsum(einsum_notation, wigner_d_3, sh_ks_deg_3), -1, -2)

        rotated_sh_ks.append(sh_ks_deg_3)

    rotated_sh_ks = np.concatenate(rotated_sh_ks, axis=-2)

    return rotated_sh_ks


def rotate_quaternions(quaternions: np.ndarray, rot_mat: np.ndarray):
    roma_quaternions = roma.quat_wxyz_to_xyzw(torch.from_numpy(quaternions).to(torch.float64))
    rot_quat = roma.rotmat_to_unitquat(torch.from_numpy(rot_mat))[None, ...]

    rotated_quaternions = roma.quat_xyzw_to_wxyz(roma.quat_product(rot_quat, roma_quaternions)).cpu().numpy()

    return rotated_quaternions


def georeference_gaussians(means, opacities, rotations, scales, sh_ks_diffuse, sh_ks_specular,
                           reference_rotation, reference_translation, reference_scaling):

    means_referenced = (np.einsum("jk,nk->nj", reference_rotation, means * reference_scaling) + reference_translation)

    scales_referenced = np.log(np.exp(scales) * reference_scaling)

    rotations_referenced = rotate_quaternions(rotations, reference_rotation)

    sh_ks_specular_referenced = rotate_sh_ks(sh_ks_specular, reference_rotation)

    return means_referenced, opacities, rotations_referenced, scales_referenced, sh_ks_diffuse, sh_ks_specular_referenced


def georeference_poses(poses, ref_rotation, ref_translation, ref_scaling):
    ref_poses = poses.copy()
    ref_poses[..., :3, 3] = np.einsum("jk,nk->nj", ref_rotation, poses[..., :3, 3] * ref_scaling) + ref_translation
    ref_poses[..., :3, :3] = ref_rotation[None, ...] @ poses[..., :3, :3]
    return ref_poses
