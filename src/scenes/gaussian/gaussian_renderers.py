import imgui
import torch
from edt_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

from src.utils.callback_utils import Callback
from src.scenes.gaussian.gaussian_models import GaussianModel
from src.utils.interfaces import RenderInfo, Render, GaussianAuxiliaryRender
from src.utils.ray_utils import depth_to_normals


class GaussianRenderer:
    def __init__(self, sh_degree=3, sh_update_frequency=1000, sh_update_start=1000,
                 progressive_low_pass=False, low_pass_update_frequency=1000, background=None, random_background=False,
                 update_statistics=True, device="cuda"):
        self.max_sh_degree = sh_degree
        self.current_sh_degree = sh_degree
        self.random_background = random_background
        self.update_statistics = update_statistics
        if background is not None:
            self.background = torch.tensor(background, dtype=torch.float32, device=device).contiguous()
        else:
            self.background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device).contiguous()

        self.sh_update_frequency = sh_update_frequency
        self.sh_update_start = sh_update_start
        self.gaussian_scale_factor = 1
        self.near = 0.05
        self.kernel_size = 0.1
        self.progressive_low_pass = progressive_low_pass
        self.low_pass_update_frequency = low_pass_update_frequency
        self.update_low_pass = False
        self.global_step = 0

    def draw_ui(self):
        imgui.text("Gaussian Renderer")
        _, self.gaussian_scale_factor = imgui.slider_float("Gaussian Scale", self.gaussian_scale_factor, 0.0, 1.0)
        _, self.current_sh_degree = imgui.slider_int("SH Degree", self.current_sh_degree, 0, 3)

    def get_callbacks(self):
        class UpdateCallback(Callback):
            def __init__(self, gaussian_renderer):
                self.gaussian_renderer = gaussian_renderer

            def on_train_begin(self):
                self.gaussian_renderer.current_sh_degree = 0

            def on_step_begin(self, global_step):
                self.gaussian_renderer.global_step = global_step
                if (global_step % self.gaussian_renderer.sh_update_frequency == 0
                        and global_step != 0
                        and global_step >= self.gaussian_renderer.sh_update_start):
                    self.gaussian_renderer.current_sh_degree = min(self.gaussian_renderer.current_sh_degree + 1,
                                                                   self.gaussian_renderer.max_sh_degree)
                if (self.gaussian_renderer.progressive_low_pass
                        and global_step % self.gaussian_renderer.low_pass_update_frequency == 0):
                    self.gaussian_renderer.update_low_pass = True

        return [UpdateCallback(self)]

    @torch.amp.autocast('cuda', dtype=torch.float32)
    def __call__(self, gaussian_model: GaussianModel, render_info: RenderInfo):
        render_info = render_info.to(torch.float32)
        perspective_transform = render_info.prespective_transform.clone()
        perspective_transform[..., 2] *= -1
        full_transform = perspective_transform @ render_info.view_transform
        view_transform = render_info.view_transform.T.contiguous()

        f, _, _, _, _ = render_info.intrinsics

        gaussian_values = gaussian_model()

        if self.update_low_pass:
            self.kernel_size = min(max(render_info.width * render_info.height /
                                       (len(gaussian_values.means) * 9 * torch.pi), 0.1), 300.0)
            self.update_low_pass = False

        subpixel_offset = torch.zeros([int(render_info.height), int(render_info.width), 2],
                                      dtype=torch.float32, device=gaussian_values.means.device)

        if self.random_background:
            background = torch.rand_like(self.background)
        else:
            background = self.background

        raster_settings = GaussianRasterizationSettings(
            image_height=render_info.height,
            image_width=render_info.width,
            tanfovx=render_info.width / (2 * f),
            tanfovy=render_info.height / (2 * f),
            kernel_size=self.kernel_size,
            subpixel_offset=subpixel_offset,
            bg=background,
            scale_modifier=self.gaussian_scale_factor,
            viewmatrix=view_transform,
            projmatrix=full_transform.T.contiguous(),
            sh_degree=self.current_sh_degree,
            campos=render_info.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings)

        means_2d = torch.zeros([gaussian_values.means.shape[0], 4],
                               dtype=gaussian_values.means.dtype,
                               requires_grad=True,
                               device=gaussian_values.means.device) + 0
        try:
            means_2d.retain_grad()
        except:
            pass

        features, depths, mean_depths, normals, opacities, distortion, radii = rasterizer(gaussian_values.means,
                                                                                          means_2d,
                                                                                          gaussian_values.opacities,
                                                                                          gaussian_values.sh_ks, None,
                                                                                          gaussian_values.scales,
                                                                                          gaussian_values.rotations,
                                                                                          None)

        features = features.moveaxis(0, -1).reshape(-1, features.shape[0])
        depths = torch.nan_to_num(depths / opacities, 0.0, 0.0, 0.0)
        depths = depths.reshape(-1)
        opacities = opacities.reshape(-1)
        distortion = distortion.reshape(-1)
        mean_depths = mean_depths.reshape(-1)
        # if not render_info.training:
        #     depths = mean_depths
        normals = normals.reshape(-1, 3)
        normals = torch.einsum("jk,nk->nj", view_transform[:3, :3], normals)
        normals = torch.nn.functional.normalize(normals)

        world_pos = None
        if render_info.ray_origins_world is not None:
            world_pos = render_info.ray_origins_world + render_info.ray_dirs_world * depths[..., None]

        render_log = {"n_gaussians_": gaussian_values.means.shape[0],
                      "sh_degree_": self.current_sh_degree,
                      "width_": render_info.width,
                      "low_pass_kernel_size_": self.kernel_size}

        depth_normals = depth_to_normals(depths, render_info, z_depth=False).reshape(-1, 3)

        auxiliary = GaussianAuxiliaryRender(gaussian_values=gaussian_values, depth_normals=depth_normals,
                                            distortion_map=distortion)

        render = Render(features, opacities, depths, logs=render_log, auxiliary=auxiliary, world_pos=world_pos,
                        normals=normals)

        return render, means_2d, radii
