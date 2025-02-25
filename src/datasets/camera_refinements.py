import torch

from src.utils.rotation_utils import axisangle_to_rot, rot_mat_from_6d, rotation_translation_to_screw_axis, \
    skrew_axis_to_rotation_translation


def randomize_poses(poses: torch.Tensor, random_rot=0.0, random_trans=0.0):
    rand_rot = torch.rand((len(poses), 3), device=poses.device) * 2 - 1
    rand_rot = rand_rot / torch.norm(rand_rot, dim=-1, keepdim=True)
    rot_deg = (torch.rand((len(poses), 1), device=poses.device) * 2 - 1) * torch.pi / 180 * random_rot
    rand_rot = rand_rot * rot_deg
    rand_rot = axisangle_to_rot(rand_rot)
    poses[:, :3, :3] = rand_rot @ poses[:, :3, :3]

    rand_trans = torch.rand((len(poses), 3), device=poses.device) * 2 - 1
    rand_trans = rand_trans * random_trans
    poses[:, :3, 3] = poses[:, :3, 3] + rand_trans

    return poses


class BasePoseRefinement(torch.nn.Module):
    def __init__(self, n_poses, egocentric=False, device="cpu"):
        super(BasePoseRefinement, self).__init__()
        self.n_poses = n_poses
        self.device = device
        self.egocentric = egocentric

        self.pose_refinements = None

        self.create_parameters()

    def create_parameters(self):
        raise NotImplementedError

    def get_new_rotation_translation(self, orig_rot, orig_trans, refinements):
        raise NotImplementedError

    def forward(self, orig_poses, idx):
        o2w = orig_poses.clone()
        orig_rot = o2w[..., :3, :3]
        orig_trans = o2w[..., :3, 3]

        if self.egocentric:
            orig_trans = torch.einsum("njk,nk->nj", -orig_rot.moveaxis(-1, -2), orig_trans)

        refinements = self.pose_refinements[idx]

        new_rot, new_trans = self.get_new_rotation_translation(orig_rot, orig_trans, refinements)

        if self.egocentric:
            new_trans = torch.einsum("njk,nk->nj", -new_rot, new_trans)

        o2w[..., :3, :3] = new_rot
        o2w[..., :3, 3] = new_trans

        return o2w


class SixDPoseRefinement(BasePoseRefinement):
    def __init__(self, *args, **kwargs):
        super(SixDPoseRefinement, self).__init__(*args, **kwargs)

    def create_parameters(self):
        self.pose_refinements = torch.nn.Parameter(torch.zeros(self.n_poses, 9).to(self.device))

    def get_new_rotation_translation(self, orig_rot, orig_trans, refinements):
        rot_refinements = refinements[..., 3:]
        trans_refinements = refinements[..., :3]
        new_rot = rot_mat_from_6d(orig_rot[..., :3, :2].moveaxis(-1, -2).reshape(-1, 6) + rot_refinements)
        new_trans = orig_trans + trans_refinements

        return new_rot, new_trans


class SE3PoseRefinement(BasePoseRefinement):
    def __init__(self, *args, **kwargs):
        super(SE3PoseRefinement, self).__init__(*args, **kwargs)

    def create_parameters(self):
        self.pose_refinements = torch.nn.Parameter(torch.zeros(self.n_poses, 6).to(self.device))

    def get_new_rotation_translation(self, orig_rot, orig_trans, refinements):
        skrew_axis = rotation_translation_to_screw_axis(orig_rot, orig_trans)

        new_skrew_axis = skrew_axis + refinements

        new_rot, new_trans = skrew_axis_to_rotation_translation(new_skrew_axis)

        return new_rot, new_trans


class IntrinsicsRefinement(torch.nn.Module):
    def __init__(self, n_cams, device="cpu", use_log_scale=True,
                 use_pix_scale=False,
                 use_principal_point=True,
                 use_distortion=True):
        super(IntrinsicsRefinement, self).__init__()
        self.n_cams = n_cams
        self.device = device

        self.use_log_scale = use_log_scale
        self.use_pix_scale = use_pix_scale
        self.use_principal_point = use_principal_point
        self.use_distortion = use_distortion

        self.focal_refinements = None
        self.principal_point_refinements = None
        self.distortion_refinements = None

        self.create_parameters()

    def create_parameters(self):
        self.focal_refinements = torch.nn.Parameter(torch.zeros(self.n_cams).to(self.device))
        if self.use_principal_point:
            self.principal_point_refinements = torch.nn.Parameter(torch.zeros(self.n_cams, 2).to(self.device))
        if self.use_distortion:
            self.distortion_refinements = torch.nn.Parameter(torch.zeros(self.n_cams, 2).to(self.device))

    def forward(self, orig_intrinsics, idx):
        f, cx, cy, k1, k2 = orig_intrinsics
        f = f.clone()
        cx = cx.clone()
        cy = cy.clone()
        k1 = k1.clone()
        k2 = k2.clone()

        pix_scale = 1.0
        if self.use_pix_scale:
            pix_scale = f.detach()

        focal_refinements = self.focal_refinements[idx]

        if self.use_log_scale:
            new_f = f * torch.exp(focal_refinements)
        else:
            new_f = f + focal_refinements * pix_scale

        if self.use_principal_point:
            principal_point_refinements = self.principal_point_refinements[idx]
            new_cx = cx + principal_point_refinements[..., 0] * pix_scale
            new_cy = cy + principal_point_refinements[..., 1] * pix_scale
        else:
            new_cx = cx
            new_cy = cy

        if self.use_distortion:
            distortion_refinements = self.distortion_refinements[idx]
            new_k1 = k1 + distortion_refinements[..., 0]
            new_k2 = k2 + distortion_refinements[..., 1]
        else:
            new_k1 = k1
            new_k2 = k2

        return new_f, new_cx, new_cy, new_k1, new_k2


base_refinement_config = {"pose_class": "SixDPoseRefinement",
                          "intrinsics_class": "IntrinsicsRefinement",
                          "egocentric": False,
                          "use_log_scale": True,
                          "use_pix_scale": False,
                          "use_principal_point": True,
                          "use_distortion": True}


class BaseCameraRefinement(torch.nn.Module):
    def __init__(self, refinement_config):
        super(BaseCameraRefinement, self).__init__()

        self.refinement_config = base_refinement_config.copy()
        self.refinement_config.update(refinement_config)

        self.config = None

        self.n_positions = None
        self.n_cams = None
        self.device = None

    def setup_parameters(self, n_positions, n_cams=1, device="cpu"):
        raise NotImplementedError

    def forward(self, orig_frame_poses, orig_alignments, orig_intrinsics, idx):
        raise NotImplementedError


class StandardCameraRefinement(BaseCameraRefinement):

    def __init__(self, refinement_config):
        super(StandardCameraRefinement, self).__init__(refinement_config)
        self.config = {"class": type(self).__name__,
                       "refinement_config": self.refinement_config}

        self.pose_refinement = None
        self.intrinsics_refinement = None
        self.alignment_refinement = None

    def setup_parameters(self, n_positions, n_cams=1, device="cpu"):
        self.n_positions = n_positions
        self.n_cams = n_cams
        self.device = device

        if self.refinement_config["pose_class"] is not None:
            self.pose_refinement = eval(self.refinement_config["pose_class"])(
                n_poses=self.n_positions,
                egocentric=self.refinement_config["egocentric"],
                device=self.device
            )

            if self.n_cams > 1:
                self.alignment_refinement = eval(self.refinement_config["pose_class"])(
                    n_poses=self.n_cams,
                    egocentric=self.refinement_config["egocentric"],
                    device=self.device
                )

        if self.refinement_config["intrinsics_class"] is not None:
            self.intrinsics_refinement = eval(self.refinement_config["intrinsics_class"])(
                n_cams=self.n_cams,
                device=self.device,
                use_log_scale=self.refinement_config["use_log_scale"],
                use_pix_scale=self.refinement_config["use_pix_scale"],
                use_principal_point=self.refinement_config["use_principal_point"],
                use_distortion=self.refinement_config["use_distortion"]
            )

    def forward(self, orig_frame_poses, orig_alignments, orig_intrinsics, idx):

        frame_idx = idx // self.n_cams
        cam_idx = idx % self.n_cams

        if self.pose_refinement is not None:
            new_frame_poses = self.pose_refinement(orig_frame_poses, frame_idx)
        else:
            new_frame_poses = orig_frame_poses

        if self.n_cams > 1 and self.alignment_refinement is not None:
            new_alignments = self.alignment_refinement(orig_alignments, cam_idx)
        else:
            new_alignments = orig_alignments

        if self.intrinsics_refinement is not None:
            new_intrinsics = self.intrinsics_refinement(orig_intrinsics, cam_idx)
        else:
            new_intrinsics = orig_intrinsics

        return new_frame_poses, new_alignments, new_intrinsics


def load_refinement(saved_config):
    refinement_class = eval(saved_config["class"])
    refinement = refinement_class(saved_config["refinement_config"])

    return refinement
