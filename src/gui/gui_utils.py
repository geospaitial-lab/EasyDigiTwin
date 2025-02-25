import warnings
from tkinter import filedialog

import cv2
import glfw
import imgui
import moderngl
import numpy as np
import torch
from moderngl.mgl import InvalidObject
from roma import euler_to_rotmat, rotmat_to_euler, rotmat_slerp

from src.datasets.cameras import TrainableCameras
from src.utils.interfaces import RenderInfo, Render, RenderMode, ProjectionType
from src.utils.image_utils import white_balance_torch, apply_gamma_correction, adjust_exposure
from src.utils.ray_utils import get_rays_orthographic, get_rays_from_pose_and_dirs, get_camera_ray_dirs, \
    get_view_transform, \
    get_perspective_transform, get_orthographic_transform
from torch2moderngl import texture2tensor, tensor2texture


def simple_button(label, callback):
    if imgui.button(label):
        callback()


class Camera:
    def __init__(self, width, height, init_angle=40, init_projection_type=ProjectionType.PERSPECTIVE, device="cuda"):
        self.width = width
        self.height = height
        self.device = device
        self.view_angle_x = np.radians(init_angle)

        self.radius = 0.5
        self.center = np.zeros(3)

        self.yaw = -90
        self.pitch = -90
        self.projection_type = init_projection_type
        self.speed = 0.05
        self.lerp_speed = 1
        self.snap_radius = 0.5

        self.interpolated_rotation = None
        self.interpolated_translation = None

        self.cx_offset = 0
        self.cy_offset = 0

        self.near = 0.002
        self.far = 10.0

    @property
    def rotation(self):
        return euler_to_rotmat("xz", [float(self.pitch), float(self.yaw)], degrees=True).numpy()

    @property
    def pose(self):
        pose = np.eye(4)
        pose[:3, :3] = self.interpolated_rotation
        pose[:3, 3] = self.interpolated_translation

        return pose

    def tick(self, delta_time):
        rot_mat = np.eye(4)
        rot_mat[:3, :3] = self.rotation
        pose = np.eye(4)
        pose[2, 3] -= self.radius
        pose = rot_mat @ pose
        pose[:3, 3] += self.center

        current_rotation = pose[:3, :3]
        current_translation = pose[:3, 3]

        lerp_speed = max(min(self.lerp_speed * delta_time * 20, 1.0), 0.0)

        self.interpolated_rotation = (rotmat_slerp(torch.tensor(self.interpolated_rotation),
                                                   torch.tensor(current_rotation),
                                                   torch.tensor(lerp_speed)).numpy()
                                      if self.interpolated_rotation is not None else current_rotation)
        self.interpolated_translation = ((1 - lerp_speed) * self.interpolated_translation
                                         + lerp_speed * current_translation
                                         if self.interpolated_translation is not None else current_translation)

    @property
    def intrinsics(self):
        cx = self.width / 2 + self.cx_offset
        cy = self.height / 2 + self.cy_offset
        f = self.width / (2 * np.tan(self.view_angle_x / 2))
        return [f, cx, cy, 0, 0]

    def orbit(self, dx, dy):
        self.yaw = self.yaw - 0.2 * dx * 2
        self.yaw = (self.yaw + 180) % 360 - 180
        self.pitch = self.pitch - 0.2 * dy * 2
        self.pitch = (self.pitch + 180) % 360 - 180

    def move_in_out(self, dr):
        self.radius *= 1.1 ** (-dr)
        if self.radius <= 0.0:
            self.radius = 0.001

    def pan(self, dx, dy, dz=0):
        self.center -= 1e-3 * self.rotation @ np.array([dx, dy, dz]) * self.radius

    def move_left_right(self, speed):
        movement = self.rotation @ np.array([1, 0, 0]) * speed
        self.center += movement

    def move_up_down(self, speed):
        movement = np.array([0, 0, 1]) * speed
        self.center += movement

    def move_front_back(self, speed):
        rotation = euler_to_rotmat("xz", [0, float(self.yaw)], degrees=True).numpy()
        movement = rotation @ np.array([0, 1, 0]) * speed
        self.center += movement

    def look_at(self, x, y, z):
        new_center = np.array([x, y, z], dtype=float)
        if np.allclose(self.pose[:3, 3], new_center):
            return
        new_relative_position = new_center - self.pose[:3, 3]
        new_radius = np.linalg.norm(new_relative_position)
        axis_sign = -np.sign(self.pitch)
        new_yaw = float(np.arctan2(axis_sign * new_relative_position[1],
                                   axis_sign * new_relative_position[0]) * 180 / np.pi) - 90
        new_pitch = float(axis_sign * (np.arcsin(new_relative_position[2] / new_radius) * 180 / np.pi - 90))

        self.radius = new_radius
        self.yaw = (new_yaw + 180) % 360 - 180
        self.pitch = (new_pitch + 180) % 360 - 180
        self.center = new_center

    def snap_to_pose(self, pose):
        rot_mat = pose[:3, :3]
        euler_angles = rotmat_to_euler("ZYX", torch.tensor(rot_mat), degrees=True).numpy()
        self.radius = self.snap_radius
        self.pitch = euler_angles[2]
        self.yaw = euler_angles[0]
        self.center = pose[:3, 3] - rot_mat @ np.array([0, 0, -self.radius])

    def resize(self, width, height):
        self.width = width
        self.height = height

    def zoom(self, new_angle):
        self.view_angle_x = new_angle

    def rays_perspective(self):
        ray_origins_world, ray_dirs_world = get_rays_from_pose_and_dirs(torch.tensor(self.pose, device=self.device,
                                                                                     dtype=torch.float32),
                                                                        get_camera_ray_dirs(self.intrinsics,
                                                                                            self.width,
                                                                                            self.height,
                                                                                            device=self.device))
        return ray_origins_world, ray_dirs_world

    def rays_orthographic(self):
        ray_origins_world, ray_dirs_world = get_rays_orthographic(torch.tensor(self.pose, device=self.device,
                                                                               dtype=torch.float32),
                                                                  self.width, self.height, self.radius, self.intrinsics)
        return ray_origins_world, ray_dirs_world

    @property
    def rays(self):
        if self.projection_type == ProjectionType.PERSPECTIVE:
            return self.rays_perspective()
        else:
            return self.rays_orthographic()

    @property
    def view_transform(self):
        camera_pose = torch.tensor(self.pose)
        view_mat = get_view_transform(camera_pose)

        return view_mat

    @property
    def perspective_transform(self):
        f, cx, cy, _, _ = self.intrinsics
        fx = torch.tensor(f)
        cx = torch.tensor(cx)
        cy = torch.tensor(cy)

        if self.projection_type == ProjectionType.PERSPECTIVE:
            projection_mat = get_perspective_transform(fx, cx, cy, self.width, self.height, self.near, self.far)
        else:
            projection_mat = get_orthographic_transform(fx, cx, cy, self.width, self.height, self.near, self.far,
                                                        self.radius)

        return projection_mat

    @property
    def camera_center(self):
        return torch.tensor(self.pose[:3, 3])

    def __call__(self):
        ray_origins_world, ray_dirs_world = self.rays
        f, cx, cy, _, _ = self.intrinsics
        intrinsics = torch.tensor([f, cx, cy, 0, 0])
        render_info = RenderInfo(ray_origins_world, ray_dirs_world,
                                 view_transform=self.view_transform,
                                 prespective_transform=self.perspective_transform,
                                 projection_type=self.projection_type,
                                 camera_center=self.camera_center,
                                 intrinsics=intrinsics,
                                 width=self.width,
                                 height=self.height)
        return render_info

    def draw_ui(self):
        with imgui.begin_menu("Camera", True) as main_menu:
            if main_menu.opened:
                imgui.text("Mode")
                if imgui.button(self.projection_type.value):
                    if self.projection_type == ProjectionType.PERSPECTIVE:
                        self.projection_type = ProjectionType.ORTHOGRAPHIC
                    else:
                        self.projection_type = ProjectionType.PERSPECTIVE

                _, self.speed = imgui.input_float("Speed", self.speed, 1e-4, 1e-2, format="%.4f",
                                                  flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                _, self.lerp_speed = imgui.slider_float("Lerp speed", self.lerp_speed, 0.01, 1.0)
                _, self.view_angle_x = imgui.slider_angle("View angle x", self.view_angle_x, 1.0, 180.0)
                _, self.cx_offset = imgui.slider_int("cx offset", self.cx_offset, -self.width // 2, self.width // 2)
                _, self.cy_offset = imgui.slider_int("cy offset", self.cy_offset, -self.height // 2, self.height // 2)
                _, self.near = imgui.slider_float("Near", self.near, 0.0001, 1.0)
                _, self.far = imgui.slider_float("Far", self.far, 1.01, 1000.0)
                _, self.snap_radius = imgui.slider_float("Snap radius", self.snap_radius, 0.0, 1.0)


class Controller:
    def __init__(self, camera, visualizer):
        self.camera = camera
        self.visualizer = visualizer

        self.pressed_keys = set()
        self.last_cursor_pos = (0, 0)
        self.last_cursor_pos_3d = (0, 0, 0)
        self.cursor_diff = [0, 0]
        self.scroll_diff = 0

        self.bindings = {
            glfw.KEY_W: lambda speed: self.camera.move_front_back(speed),
            glfw.KEY_S: lambda speed: self.camera.move_front_back(-speed),
            glfw.KEY_A: lambda speed: self.camera.move_left_right(-speed),
            glfw.KEY_D: lambda speed: self.camera.move_left_right(speed),
            glfw.KEY_Q: lambda speed: self.camera.move_up_down(speed),
            glfw.KEY_E: lambda speed: self.camera.move_up_down(-speed),
            glfw.KEY_O: lambda speed: self.camera.look_at(0, 0, 0),
            glfw.KEY_1: lambda speed: setattr(self.visualizer, 'render_mode', RenderMode.RGB),
            glfw.KEY_2: lambda speed: setattr(self.visualizer, 'render_mode', RenderMode.DEPTH),
            glfw.KEY_3: lambda speed: setattr(self.visualizer, 'render_mode', RenderMode.OPACITY),
            glfw.KEY_4: lambda speed: setattr(self.visualizer, 'render_mode', RenderMode.NORMAL),
            glfw.KEY_5: lambda speed: setattr(self.visualizer, 'render_mode', RenderMode.DEPTH_NORMAL),
            glfw.KEY_6: lambda speed: setattr(self.visualizer, 'render_mode', RenderMode.WORLD_POS),
            glfw.MOUSE_BUTTON_LEFT: lambda speed: self.camera.orbit(self.cursor_diff[0], self.cursor_diff[1]),
            glfw.MOUSE_BUTTON_RIGHT: lambda speed: self.camera.pan(self.cursor_diff[0], self.cursor_diff[1]),
            glfw.MOUSE_BUTTON_MIDDLE: lambda speed: setattr(self.camera, "center", np.array(self.last_cursor_pos_3d)),
            glfw.MOUSE_BUTTON_4: lambda speed: self.camera.look_at(self.last_cursor_pos_3d[0],
                                                                   self.last_cursor_pos_3d[1],
                                                                   self.last_cursor_pos_3d[2]),
            "scroll": lambda speed: self.camera.move_in_out(self.scroll_diff)
        }

        self.overrides = {
            glfw.KEY_W: [glfw.KEY_S],
            glfw.KEY_A: [glfw.KEY_D],
            glfw.KEY_Q: [glfw.KEY_E],
            glfw.KEY_1: [glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5, glfw.KEY_6],
            glfw.KEY_2: [glfw.KEY_3, glfw.KEY_4, glfw.KEY_5, glfw.KEY_6],
            glfw.KEY_3: [glfw.KEY_4, glfw.KEY_5, glfw.KEY_6],
            glfw.KEY_4: [glfw.KEY_5, glfw.KEY_6],
            glfw.KEY_5: [glfw.KEY_6],
        }

        self.single_action = [
            glfw.MOUSE_BUTTON_MIDDLE,
            glfw.MOUSE_BUTTON_4,
            "scroll"
        ]

    def key_callback(self, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.pressed_keys.add(key)
        elif action == glfw.RELEASE:
            self.pressed_keys.discard(key)

    def mouse_button_callback(self, button, action, mods):
        if action == glfw.PRESS:
            self.pressed_keys.add(button)
        elif action == glfw.RELEASE:
            self.pressed_keys.discard(button)

    def cursor_pos_callback(self, x, y):
        self.cursor_diff[0] += x - self.last_cursor_pos[0]
        self.cursor_diff[1] += y - self.last_cursor_pos[1]
        self.last_cursor_pos = (x, y)

    def cursor_pos_3d_callback(self, x, y, z):
        self.last_cursor_pos_3d = (x, y, z)

    def scroll_callback(self, xoffset, yoffset):
        self.scroll_diff += yoffset
        self.pressed_keys.add("scroll")

    def parse_keys(self, pressed_keys):
        parsed_keys = pressed_keys.copy()
        for key, overrides in self.overrides.items():
            if key in pressed_keys:
                for override in overrides:
                    parsed_keys.discard(override)

        return parsed_keys

    def clean_up_keys(self, pressed_keys):
        for key in self.single_action:
            pressed_keys.discard(key)

    def tick(self, delta_time):
        speed = self.camera.speed * delta_time * 20
        parsed_keys = self.parse_keys(self.pressed_keys)

        for key in parsed_keys:
            if key in self.bindings:
                self.bindings[key](speed)

        self.cursor_diff = [0, 0]
        self.scroll_diff = 0

        self.clean_up_keys(self.pressed_keys)


class Recorder:
    def __init__(self, camera, visualizer, initial_dir=None):
        self.camera = camera
        self.visualizer = visualizer
        self.initial_dir = initial_dir

        self.pose_list = []
        self._current_pose_id = -1
        self._current_progress = 0
        self.travel_time = 2
        self.playing = False

        self.video_writer = None
        self.wants_to_record = False

        self.recording_width = 2448
        self.recording_height = 2048
        self.recording_fps = 30

    @property
    def current_pose_id(self):
        return self._current_pose_id

    @current_pose_id.setter
    def current_pose_id(self, value):
        self._current_pose_id = min(max(0, value), len(self.pose_list) - 1)

    @property
    def current_progress(self):
        return self._current_progress

    @current_progress.setter
    def current_progress(self, value):
        self._current_progress = min(max(0.0, value), 1.0)

    def add_after_pose(self):
        self.pose_list.insert(self.current_pose_id + 1, self.camera.pose)
        self.current_pose_id += 1

    def remove_current_pose(self):
        self.pose_list.pop(self.current_pose_id)

    def save_poses(self):
        file_path = filedialog.asksaveasfilename(filetypes=[("NPY files", "*.npy")],
                                                 initialdir=self.initial_dir)
        if file_path:
            np.save(file_path, np.stack(self.pose_list))

    def load_poses(self):
        file_path = filedialog.askopenfilename(filetypes=[("NPY files", "*.npy")],
                                               initialdir=self.initial_dir)
        if file_path:
            self.pose_list = [pose for pose in np.load(file_path)]

    def start_playback(self):
        if len(self.pose_list) <= 1:
            warnings.warn("Not enough recorded poses to play")
            return
        self.current_pose_id = 0
        self.current_progress = 0.0
        self.playing = True

    def stop_playback(self):
        self.current_progress = 0.0
        self.playing = False

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def create_writer(self):
        file_path = filedialog.asksaveasfilename(filetypes=[("Video files", "*.mp4")],
                                                 initialdir=self.initial_dir)
        if file_path:
            self.video_writer = cv2.VideoWriter(file_path,
                                                cv2.VideoWriter_fourcc(*'mp4v'),
                                                self.recording_fps,
                                                (self.recording_width, self.recording_height))

    def start_recording(self):
        self.start_playback()
        if self.playing:
            self.create_writer()
        self.wants_to_record = False

    def write_frame(self, image):
        if self.video_writer is None:
            warnings.warn("No video writer!")
            self.stop_playback()
            return
        self.video_writer.write(cv2.cvtColor((image.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    def tick(self, delta_time):
        if self.playing:
            rotation = rotmat_slerp(torch.tensor(self.pose_list[self.current_pose_id][:3, :3]),
                                    torch.tensor(self.pose_list[self.current_pose_id + 1][:3, :3]),
                                    torch.tensor(self.current_progress)).numpy()
            translation = ((1 - self.current_progress) * self.pose_list[self.current_pose_id][:3, 3]
                           + self.current_progress * self.pose_list[self.current_pose_id + 1][:3, 3])

            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = translation

            self.camera.snap_to_pose(pose)

            if self.current_progress >= 1.0:
                if self.current_pose_id == len(self.pose_list) - 2:
                    self.stop_playback()
                else:
                    self.current_progress = 0.0
                    self.current_pose_id += 1

            segment_length = np.linalg.norm(self.pose_list[self.current_pose_id + 1][:3, 3]
                                            - self.pose_list[self.current_pose_id][:3, 3])

            self.current_progress += min(delta_time / (self.travel_time * segment_length), 1.0)

    def draw_ui(self):
        with imgui.begin_menu("Recording", True) as main_menu:
            if main_menu.opened:
                simple_button("Add pose", self.add_after_pose)
                imgui.same_line()
                imgui.text(f"{len(self.pose_list)} currently poses")

                simple_button("Remove pose", self.remove_current_pose)

                imgui.same_line()

                simple_button("Clear poses",  self.pose_list.clear)

                simple_button("Save poses", self.save_poses)

                imgui.same_line()

                simple_button("Load poses", self.load_poses)

                recording_pose_changed, self.current_pose_id = imgui.input_int("Current pose",
                                                                               self.current_pose_id, 1, 10,
                                                                               flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)

                if recording_pose_changed and len(self.pose_list) > 0:
                    self.camera.snap_to_pose(self.pose_list[self.current_pose_id])

                _, self.travel_time = imgui.slider_float("Travel time", self.travel_time,0.1, 5.0)

                if self.playing:
                    simple_button("Stop playback", lambda: setattr(self, "playing", False))
                else:
                    simple_button("Playback poses", self.start_playback)

                with imgui.begin_menu("Recording options", True) as recording_menu:
                    if recording_menu.opened:
                        _, self.recording_width = imgui.input_int("Recording width", self.recording_width, 1, 100)
                        _, self.recording_height = imgui.input_int("Recording height", self.recording_height, 1, 100)
                        _, self.recording_fps = imgui.input_int("Recording fps", self.recording_fps, 1, 10)

                simple_button("Start recording", lambda: setattr(self, "wants_to_record", True))


def axis2gl(perspective_transform):

    perspective_transform = torch.clone(perspective_transform)
    perspective_transform[:, 1] *= -1
    perspective_transform[:, 2] *= -1

    return perspective_transform


def set_program_value(program, key, value):
    """

    :param moderngl.Program program:
    :param str key:
    :param value:
    :return:
    """
    if key in program.__dict__["_members"]:
        program[key] = value
    else:
        warnings.warn(f"{key} not in {program}!")


class GLRenderer:
    def __init__(self, width, height):
        """
        Renderer responsible for rendering OpenGL objects

        :param int width: initial width
        :param int height: initial height
        """

        self.camera_center = np.zeros(3)

        # GL Context
        self.ctx = moderngl.create_context()
        self.ctx.gc_mode = "context_gc"
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self.gl_objects = {}

        # Shader
        with open("src/gui/shaders/simple.vert") as file:
            vertex_shader = file.read()
        with open("src/gui/shaders/simple.frag") as file:
            fragment_shader = file.read()
        self.program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self.gl_objects["program"] = self.program

        with open("src/gui/shaders/texture2screen.vert") as file:
            vertex_shader = file.read()
        with open("src/gui/shaders/texture2screen.frag") as file:
            fragment_shader = file.read()
        self.screen_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        set_program_value(self.screen_program, "tex", 2)
        self.gl_objects["screen_program"] = self.screen_program

        # Vertex Arrays
        self.center_vao = None
        self.gl_objects["center_vao"] = self.center_vao
        self.cam_vao = None
        self.gl_objects["cam_vao"] = self.cam_vao
        self.gt_cam_vao = None
        self.gl_objects["gt_cam_vao"] = self.gt_cam_vao
        self.orig_cam_vao = None
        self.gl_objects["orig_cam_vao"] = self.orig_cam_vao
        self.gt_to_train_cam_vao = None
        self.gl_objects["gt_to_train_cam_vao"] = self.gt_to_train_cam_vao
        self.orig_to_train_cam_vao = None
        self.gl_objects["orig_to_train_cam_vao"] = self.orig_to_train_cam_vao
        self.origin_vao = None
        self.gl_objects["origin_vao"] = self.origin_vao
        self.bounds_vao = None
        self.gl_objects["bounds_vao"] = self.bounds_vao
        self.quad_vao = None
        self.gl_objects["quad_vao"] = self.quad_vao

        # Show flags
        self.show_center = False
        self.show_cam_poses = False
        self.show_gt_cam_poses = False
        self.show_orig_cam_poses = False
        self.show_origin = False
        self.show_bounds = False

        self.cam_scale = 0.01
        self.center_scale = 0.01

        # Containers for Synchronous update of vertices
        self.new_train_cams = None
        self.current_train_cams = None
        self.new_gt_cams = None
        self.new_orig_cams = None
        self.num_cams = 0
        self.new_roi = None

        # Render Buffer
        self.default_fbo = self.ctx.fbo

        self.color_texture = None
        self.pos_depth_texture = None
        self.screen_texture = None
        self.depth_buffer = None

        self.fbo = None

        self.resize_buffers(width, height)

        # Vertex setup
        self.create_origin_vertex_array()
        self.create_camera_center_vertices()
        self.create_quad_vertex_array()

    def resize_buffers(self, width, height):
        if self.color_texture is not None and not isinstance(self.color_texture.mglo, InvalidObject):
            self.color_texture.release()
        self.color_texture = self.ctx.texture((width, height), 4)
        self.color_texture.use(location=0)
        self.gl_objects["color_texture"] = self.color_texture

        if self.pos_depth_texture is not None and not isinstance(self.pos_depth_texture.mglo, InvalidObject):
            self.pos_depth_texture.release()
        self.pos_depth_texture = self.ctx.texture((width, height), 4, dtype="f4")
        self.pos_depth_texture.use(location=1)
        self.gl_objects["pos_depth_texture"] = self.pos_depth_texture

        if self.screen_texture is not None and not isinstance(self.screen_texture.mglo, InvalidObject):
            self.screen_texture.release()
        self.screen_texture = self.ctx.texture((width, height), 4)
        self.screen_texture.use(location=2)
        self.gl_objects["screen_texture"] = self.screen_texture

        if self.depth_buffer is not None and not isinstance(self.depth_buffer.mglo, InvalidObject):
            self.depth_buffer.release()
        self.depth_buffer = self.ctx.depth_renderbuffer((width, height))
        self.gl_objects["depth_buffer"] = self.depth_buffer

        if self.fbo is not None and not isinstance(self.fbo.mglo, InvalidObject):
            self.fbo.release()
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color_texture, self.pos_depth_texture],
                                        depth_attachment=self.depth_buffer)
        self.gl_objects["fbo"] = self.fbo

    def resize(self, width, height):
        self.resize_buffers(width, height)
        self.ctx.viewport = (0, 0, width, height)

    def destroy_global_object(self, object_name):
        if self.gl_objects[object_name] is not None and not isinstance(self.gl_objects[object_name], InvalidObject):
            self.gl_objects[object_name].release()
            setattr(self, object_name, None)
            self.gl_objects[object_name] = getattr(self, object_name)

    def destroy_context(self):
        for gl_object in self.gl_objects.values():
            if gl_object is not None and not isinstance(gl_object, InvalidObject):
                gl_object.release()

    def delete_train_cams(self):
        self.destroy_global_object("cam_vao")
        self.destroy_global_object("gt_cam_vao")
        self.destroy_global_object("orig_cam_vao")
        self.destroy_global_object("orig_to_train_cam_vao")
        self.destroy_global_object("gt_to_train_cam_vao")
        self.num_cams = 0
        self.current_train_cams = None
        self.new_train_cams = None
        self.show_cam_poses = False
        self.show_gt_cam_poses = False
        self.show_orig_cam_poses = False

    def update_train_cams(self, cameras: TrainableCameras | None):
        if cameras is None:
            self.delete_train_cams()
        else:
            self.new_train_cams = cameras

    def update_bounds(self, roi):
        self.new_roi = roi

    def update_cam_scale(self, new_scale):
        self.cam_scale = new_scale
        if self.new_train_cams is None:
            self.new_train_cams = self.current_train_cams

    def update_center_scale(self, new_scale):
        self.center_scale = new_scale
        self.create_camera_center_vertices()

    def create_vertex_array(self, vertices, edges, poses=None):
        vbo = self.ctx.buffer(vertices.astype("f4"))
        ibo = self.ctx.buffer(edges.astype("i4"))

        if poses is not None:
            position_buffer = self.ctx.buffer(np.ascontiguousarray(np.transpose(poses, axes=[0, 2, 1]).astype("f4")))
        else:
            position_buffer = self.ctx.buffer(np.eye(4).astype("f4"))
        vao_content = [
            (vbo, "3f 3f", "in_vert", "in_color"),
            (position_buffer, "16f /i", "in_pose")

        ]
        vao = self.ctx.vertex_array(self.program, vao_content, index_buffer=ibo)

        return vao

    def create_camera_vertex_array(self, intrinsics, poses, target_vao_name, color=None):
        if color is None:
            color = [0, 0, 0]
        if getattr(self, target_vao_name) is not None \
                and type(getattr(self, target_vao_name).mglo) != moderngl.mgl.InvalidObject:
            getattr(self, target_vao_name).release()

        f, cx, cy, _, _ = intrinsics

        cam_scale = self.cam_scale
        cx = cx / f * cam_scale
        cy = cy / f * cam_scale
        fx = cam_scale

        vertices = np.array([[0, 0, 0] + color, [-cx, -cy, fx] + color, [cx, -cy, fx] + color,
                             [cx, cy, fx] + color, [-cx, cy, fx] + color])

        edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])

        cam_vao = self.create_vertex_array(vertices, edges, poses)
        setattr(self, target_vao_name, cam_vao)
        self.gl_objects[target_vao_name] = cam_vao

    def create_camera_correspondence_vertex_array(self, cam_1_poses, cam_2_poses, target_vao_name, color=None):
        if color is None:
            color = [255, 0, 255]
        if getattr(self, target_vao_name) is not None \
                and type(getattr(self, target_vao_name).mglo) != moderngl.mgl.InvalidObject:
            getattr(self, target_vao_name).release()

        cam_1_coords = cam_1_poses[..., :3, 3]
        cam_2_coords = cam_2_poses[..., :3, 3]

        colors = np.ones_like(cam_1_coords) * np.array([color])
        cam_1_coords = np.concatenate([cam_1_coords, colors], axis=-1)
        cam_2_coords = np.concatenate([cam_2_coords, colors], axis=-1)

        vertices = np.concatenate([cam_1_coords, cam_2_coords], axis=0)

        edges = np.array([[i, i + len(cam_1_coords)] for i in range(len(cam_1_coords))])

        correspondence_vao = self.create_vertex_array(vertices, edges)
        setattr(self, target_vao_name, correspondence_vao)
        self.gl_objects[target_vao_name] = correspondence_vao

    def create_origin_vertex_array(self):
        if self.origin_vao is not None and not isinstance(self.origin_vao.mglo, InvalidObject):
            self.origin_vao.release()

        vertices = np.array([[0, 0, 0, 1, 0, 0], [0.1, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0], [0, 0.1, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1], [0, 0, 0.1, 0, 0, 1]])

        edges = np.array([[0, 1], [2, 3], [4, 5]])

        self.origin_vao = self.create_vertex_array(vertices, edges)
        self.gl_objects["origin_vao"] = self.origin_vao

    def create_bounds_vertex_array(self, roi):
        if self.bounds_vao is not None and not isinstance(self.bounds_vao.mglo, InvalidObject):
            self.bounds_vao.release()

        xmin, ymin, zmin, xmax, ymax, zmax = roi

        vertices = np.array([[xmin, ymin, zmin, 0, 0, 0], [xmax, ymin, zmin, 0, 0, 0], [xmax, ymax, zmin, 0, 0, 0],
                             [xmin, ymax, zmin, 0, 0, 0], [xmin, ymin, zmax, 0, 0, 0], [xmax, ymin, zmax, 0, 0, 0],
                             [xmax, ymax, zmax, 0, 0, 0], [xmin, ymax, zmax, 0, 0, 0]])

        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]])

        self.bounds_vao = self.create_vertex_array(vertices, edges)
        self.gl_objects["bounds_vao"] = self.bounds_vao

    def create_quad_vertex_array(self):
        vertices = np.array([[-1, 1, 0, 0.0, 1.0],
                             [-1, -1, 0, 0.0, 0.0],
                             [1, -1, 0, 1.0, 0.0],
                             [-1, 1, 0, 0.0, 1.0],
                             [1, -1, 0, 1.0, 0.0],
                             [1, 1, 0, 1.0, 1.0]])

        edges = np.array([[0, 1], [2, 3], [4, 5]])

        vbo = self.ctx.buffer(vertices.astype("f4"))
        ibo = self.ctx.buffer(edges.astype("i4"))

        vao_content = [
            (vbo, "3f 2f", "in_vert", "in_uv")
        ]
        self.quad_vao = self.ctx.vertex_array(self.screen_program, vao_content, index_buffer=ibo)
        self.gl_objects["quad_vao"] = self.quad_vao

    def create_camera_center_vertices(self):
        if self.center_vao is not None and not isinstance(self.center_vao.mglo, InvalidObject):
            self.center_vao.release()

        vertices = np.array([[self.center_scale, 0, 0, 1, 0, 0], [0, self.center_scale, 0, 0, 1, 0],
                             [-self.center_scale, 0, 0, 0, 0, 0], [0, -self.center_scale, 0, 0, 0, 0],
                             [0, 0, self.center_scale, 0, 0, 1], [0, 0, -self.center_scale, 0, 0, 0]])

        edges = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 5, 1], [1, 5, 2], [2, 5, 3], [3, 5, 0]])

        self.center_vao = self.create_vertex_array(vertices, edges)
        self.gl_objects["center_vao"] = self.center_vao

    def render_vao(self, vao, primitive_mode=moderngl.LINES, offset=np.array([0, 0, 0]), instances=-1):
        if vao is None:
            return
        set_program_value(self.program, "offset", offset)

        vao.render(primitive_mode, instances=instances)

    def update_vertices(self):
        if self.new_train_cams is not None:
            intrinsics = [v.cpu().numpy() for v in self.new_train_cams.intrinsics(torch.tensor([0]))]
            poses = self.new_train_cams.poses(...).cpu().numpy()
            orig_poses = self.new_train_cams.orig_poses.cpu().numpy()
            gt_poses = self.new_train_cams.gt_poses.cpu().numpy()
            self.create_camera_vertex_array(intrinsics, poses, "cam_vao")
            self.create_camera_vertex_array(intrinsics, gt_poses, "gt_cam_vao", color=[0, 255, 0])
            self.create_camera_vertex_array(intrinsics, orig_poses, "orig_cam_vao", color=[255, 0, 0])
            self.create_camera_correspondence_vertex_array(poses, orig_poses, "orig_to_train_cam_vao")
            self.create_camera_correspondence_vertex_array(poses, gt_poses, "gt_to_train_cam_vao")
            self.num_cams = len(self.new_train_cams)
            self.current_train_cams = self.new_train_cams
            self.new_train_cams = None
        if self.new_roi is not None:
            self.create_bounds_vertex_array(self.new_roi)
            self.new_roi = None

    def render(self, render_info: RenderInfo):
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.update_vertices()

        view_mat = render_info.view_transform.cpu().numpy()
        projection_mat = axis2gl(render_info.prespective_transform).cpu().numpy()
        view_pos = render_info.camera_center.cpu().numpy()

        set_program_value(self.program, "w2c", np.reshape(view_mat.T, -1))
        set_program_value(self.program, "projection", np.reshape(projection_mat.T, -1))
        set_program_value(self.program, "view_pos", view_pos)
        set_program_value(self.program, "perspective", render_info.projection_type == ProjectionType.PERSPECTIVE)

        if self.show_cam_poses:
            self.render_vao(self.cam_vao, instances=self.num_cams)
        if self.show_gt_cam_poses:
            self.render_vao(self.gt_cam_vao, instances=self.num_cams)
        if self.show_cam_poses and self.show_gt_cam_poses:
            self.render_vao(self.gt_to_train_cam_vao, instances=self.num_cams)
        if self.show_orig_cam_poses:
            self.render_vao(self.orig_cam_vao, instances=self.num_cams)
        if self.show_cam_poses and self.show_orig_cam_poses:
            self.render_vao(self.orig_to_train_cam_vao, instances=self.num_cams)
        if self.show_origin:
            self.render_vao(self.origin_vao)
        if self.show_bounds:
            self.render_vao(self.bounds_vao)
        if self.show_center:
            self.render_vao(self.center_vao, offset=self.camera_center, primitive_mode=moderngl.TRIANGLES)

        image_tensor = texture2tensor(self.color_texture)

        image_tensor = image_tensor[..., :3] / 255

        pos_depth_tensor = texture2tensor(self.pos_depth_texture)

        self.default_fbo.use()

        image_tensor = torch.reshape(image_tensor, (-1, 3))
        world_pos = torch.reshape(pos_depth_tensor[..., :3], (-1, 3))
        depths = torch.reshape(pos_depth_tensor[..., -1], (-1,))
        opacities = torch.ones_like(depths)

        gl_render = Render(image_tensor, opacities, depths, world_pos=world_pos)

        return gl_render

    def tensor_to_screen(self, image_tensor):
        tensor2texture(image_tensor, self.screen_texture)
        self.default_fbo.use()
        self.default_fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.quad_vao.render(moderngl.TRIANGLES)

    def draw_ui(self):
        with imgui.begin_menu("GL Overlay", True) as main_menu:
            if main_menu.opened:
                _, self.show_center = imgui.checkbox("Camera Center", self.show_center)
                if self.num_cams > 0:
                    _, self.show_cam_poses = imgui.checkbox("Camera Poses", self.show_cam_poses)
                    _, self.show_gt_cam_poses = imgui.checkbox("Ground Truth Camera Poses",
                                                               self.show_gt_cam_poses)
                    _, self.show_orig_cam_poses = imgui.checkbox("Original Camera Poses",
                                                                 self.show_orig_cam_poses)
                _, self.show_origin = imgui.checkbox("Origin", self.show_origin)
                _, self.show_bounds = imgui.checkbox("Roi", self.show_bounds)

                cam_scale_changed, cam_scale = imgui.slider_float("Camera Scale", self.cam_scale, 0.0, 1.0)
                if cam_scale_changed:
                    self.update_cam_scale(cam_scale)

                center_scale_changed, center_scale = imgui.slider_float("Center Scale", self.center_scale, 0.0001, 0.1)
                if center_scale_changed:
                    self.update_center_scale(center_scale)

                imgui.text(f"Camera center: {self.camera_center}")


class Visualizer:
    def __init__(self):
        self._render_mode = RenderMode.RGB
        self.possible_modes = [RenderMode.RGB]

        self.depth_color_near = torch.tensor([0.0, 0.0, 0.0])
        self.depth_color_far = torch.tensor([1.0, 1.0, 1.0])

        self.white_balance_factors = np.array([1.0, 1.0, 1.0])
        self.gamma = 1.0
        self.exposure = 1.0

    @property
    def render_mode(self):
        return self._render_mode

    @render_mode.setter
    def render_mode(self, new_render_mode):
        if new_render_mode in self.possible_modes:
            self._render_mode = new_render_mode

    def __call__(self, render, gl_render):
        self.update_modes(render)
        depths = render.depths
        gl_depths = gl_render.depths

        if self._render_mode == RenderMode.RGB:
            rgbs = render.features
            rgbs = white_balance_torch(rgbs, self.white_balance_factors)
            rgbs = apply_gamma_correction(rgbs, self.gamma)
            rgbs = adjust_exposure(rgbs, factor=self.exposure)
            rgbs = torch.clamp(rgbs, 0, 1)
            gl_rgbs = torch.clamp(gl_render.features, 0, 1)
        elif self._render_mode == RenderMode.DEPTH:
            rgbs, gl_rgbs = self.depth_to_color(render.depths, gl_render.depths)
            rgbs = torch.clamp(rgbs, 0, 1)
            gl_rgbs = torch.clamp(gl_rgbs, 0, 1)
        elif self._render_mode == RenderMode.OPACITY:
            rgbs = torch.clamp(self.opacity_to_color(render.opacities), 0, 1)
            gl_rgbs = torch.clamp(self.opacity_to_color(gl_render.opacities), 0, 1)
        elif self._render_mode == RenderMode.NORMAL:
            rgbs = torch.clamp(self.normal_to_color(render.normals), 0, 1)
            gl_rgbs = torch.zeros_like(gl_render.features)
        elif self._render_mode == RenderMode.DEPTH_NORMAL:
            rgbs = torch.clamp(self.normal_to_color(render.auxiliary.depth_normals), 0, 1)
            gl_rgbs = torch.zeros_like(gl_render.features)
        elif self._render_mode == RenderMode.WORLD_POS:
            rgbs = torch.clamp(render.world_pos, 0, 1)
            gl_rgbs = torch.clamp(gl_render.world_pos, 0, 1)
        elif self._render_mode == RenderMode.DISTORTION:
            rgbs, _ = self.depth_to_color(render.auxiliary.distortion_map, None)
            gl_rgbs = torch.clamp(gl_render.features, 0, 1)
        else:
            raise NotImplementedError

        final_rgbs = self.composit(rgbs, gl_rgbs, depths, gl_depths)

        return final_rgbs

    def composit(self, rgbs, gl_rgbs, depths, gl_depths):
        condition = torch.logical_and(torch.logical_or(gl_depths < depths, depths == 0), gl_depths != 0)
        return torch.where(condition[..., None].expand(rgbs.size()), gl_rgbs, rgbs)

    def update_modes(self, render):
        modes = [RenderMode.RGB]
        if render.depths is not None:
            modes.append(RenderMode.DEPTH)
        if render.opacities is not None:
            modes.append(RenderMode.OPACITY)
        if render.normals is not None:
            modes.append(RenderMode.NORMAL)
        if render.auxiliary is not None and render.auxiliary.depth_normals is not None:
            modes.append(RenderMode.DEPTH_NORMAL)
        if render.world_pos is not None:
            modes.append(RenderMode.WORLD_POS)
        if render.auxiliary is not None and render.auxiliary.distortion_map is not None:
            modes.append(RenderMode.DISTORTION)

        self.possible_modes = modes

    def depth_to_color(self, depth, gl_depth=None):
        background_idx = torch.where(depth == 0)
        depth[background_idx] = depth.max() * 1.25
        max_depth = depth.max()
        min_depth = depth.min()
        gl_depth_image = None
        if gl_depth is not None:
            gl_background_idx = torch.where(gl_depth == 0)
            gl_depth[gl_background_idx] = gl_depth.max() * 1.25
            max_depth = max(max_depth, gl_depth.max())
            min_depth = min(min_depth, gl_depth.min())

            gl_depth = (gl_depth - min_depth) / (max_depth - min_depth)
            gl_depth_image = gl_depth[..., None].repeat(1, 1, 3)

            gl_depth_image = (gl_depth_image * self.depth_color_far.to(gl_depth_image.device)
                              + (1 - gl_depth_image) * self.depth_color_near.to(gl_depth_image.device))

        depth = (depth - min_depth) / (max_depth - min_depth)
        depth_image = depth[..., None].repeat(1, 1, 3)

        depth_image = (depth_image * self.depth_color_far.to(depth_image.device)
                       + (1 - depth_image) * self.depth_color_near.to(depth_image.device))

        return depth_image, gl_depth_image

    def opacity_to_color(self, opacities):
        opacity_image = opacities[..., None].repeat(1, 1, 3)

        return opacity_image

    def normal_to_color(self, normals):
        normal_image = (normals + 1) / 2

        return normal_image

    def draw_ui(self):
        with imgui.begin_menu("Visualization", True) as main_menu:
            if main_menu.opened:
                with imgui.begin_combo("Render mode", self._render_mode.value) as combo:
                    if combo.opened:
                        for i, item in enumerate(self.possible_modes):
                            is_selected = (item == self._render_mode)
                            if imgui.selectable(item.value, is_selected)[0]:
                                self._render_mode = item

                            # Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                            if is_selected:
                                imgui.set_item_default_focus()

                (_,
                 (self.white_balance_factors[0],
                  self.white_balance_factors[1],
                  self.white_balance_factors[2])) = imgui.input_float3("White balance",
                                                                       self.white_balance_factors[0],
                                                                       self.white_balance_factors[1],
                                                                       self.white_balance_factors[2],
                                                                       format="%.6f",
                                                                       flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                _, self.gamma = imgui.input_float("Gamma",
                                                  self.gamma,
                                                  format="%.2f",
                                                  flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                _, self.exposure = imgui.input_float("Exposure",
                                                     self.exposure,
                                                     format="%.2f",
                                                     flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
