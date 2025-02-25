import os
import sys
import tkinter as tk
from os.path import dirname
from tkinter import filedialog

import numpy as np

from src.scenes.scene_model import EmptyScene
from src.scripts.train import get_trainer
from src.utils.run_utils import load_scene, load_pointcloud, load_empty_scene

sys.path.append(os.path.abspath('..'))

import glfw
import imgui
import torch
from imgui.integrations.glfw import GlfwRenderer

from src.gui.gui_utils import Camera, GLRenderer, Visualizer, Controller, Recorder, simple_button
from src.utils.interfaces import Render, SnapMode

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
root = tk.Tk()
root.withdraw()


class GUI:
    @torch.no_grad()
    def __init__(self, scene_path=None, width=800, height=800, init_angle=40, device="cuda", trainer=None):
        self.device = device
        self.width = width
        self.height = height
        self.original_width = width
        self.original_height = height

        self.trainer = trainer
        self.save_trainer = False
        self.train_steps = 1
        self.train_continuous = False

        self.scene_path = scene_path
        self.scene, self.train_cams, self.geo_reference = load_empty_scene(device=self.device)

        self.camera = Camera(width, height, init_angle=init_angle, device=self.device)
        self.visualizer = Visualizer()
        self.controller: Controller|None = None
        self.recorder = None
        self.recording = False

        self.last_cursor_pos = (0, 0)

        self.snap_pose_id = 0
        self.snap_mode = SnapMode.SNAP

        self.window = None
        self.glfw_renderer = None
        self.current_fps = 0
        self.pressed_keys = set()

        self.setup_gui()

        self.gl_renderer = GLRenderer(self.width, self.height)

        if self.trainer is not None:
            self.load_trainer(self.trainer)
        elif scene_path is not None:
            self.load_from_path(self.scene_path)

    def get_initial_dir(self):
        if self.scene_path is None:
            return None
        if os.path.isfile(self.scene_path):
            initial_dir = dirname(dirname(self.scene_path))
        else:
            initial_dir = dirname(self.scene_path)

        return initial_dir

    def load_trainer(self, trainer):
        self.trainer = trainer
        self.scene = self.trainer.scene_model
        self.train_cams = self.trainer.train_dataset.cameras
        self.geo_reference = None

        self.gl_renderer.update_bounds(self.scene.model.get_roi())
        self.gl_renderer.update_train_cams(self.train_cams)

    def load_scene(self, scene_path):
        self.scene_path = scene_path
        self.scene, self.train_cams, self.geo_reference = load_scene(scene_path, device=self.device)

    def load_pointcloud(self, ply_path):
        self.scene_path = ply_path
        self.scene, self.train_cams, self.geo_reference = load_pointcloud(ply_path, device=self.device)

    def load_from_path(self, scene_path):
        if scene_path.endswith("scene_model.json"):
            self.load_scene(scene_path)
        elif scene_path.endswith("ply") or scene_path.endswith("las") or scene_path.endswith("laz"):
            self.load_pointcloud(scene_path)

        self.gl_renderer.update_bounds(self.scene.model.get_roi())
        self.gl_renderer.update_train_cams(self.train_cams)

        if scene_path.endswith(".las") or scene_path.endswith(".laz"):
            self.camera.far = 200.0
            self.camera.speed = 0.3
            self.gl_renderer.update_cam_scale(0.2)
            self.gl_renderer.update_center_scale(0.1)
        else:
            self.camera.far = 10.0
            self.camera.speed = 0.05
            self.gl_renderer.update_cam_scale(0.01)
            self.gl_renderer.update_center_scale(0.01)

        if not isinstance(self.scene.model, EmptyScene):
            scene_center = self.scene.model.model().means.mean(dim=0)
            self.camera.center = scene_center.cpu().numpy()

    def load_button_callback(self):
        file_path = filedialog.askopenfilename(initialdir=self.get_initial_dir(),
                                               filetypes=[("Supported Files", "scene_model.json *.ply *.las *.laz"),
                                                          ("Scene", "scene_model.json"),
                                                          ("PLY", "*.ply"),
                                                          ("LAS/LAZ", "*.las *.laz"),
                                                          ("All Files", "*.*")]
                                               )

        self.load_from_path(file_path)

    def measure_button_callback(self):
        coords = (np.einsum("jk,k->j",
                            self.geo_reference["rotation"],
                            self.camera.center * self.geo_reference["scaling"])
                  + self.geo_reference["translation"])
        coors_string = f"{coords[0]}\t{coords[1]}\t{coords[2]}".replace(".", ",")
        print(coors_string)
        os.system(f"echo {coors_string} | clip")

    def setup_gui(self):
        imgui.create_context()
        glfw.init()

        self.window = glfw.create_window(self.width, self.height, "EasyDigiTwin", None, None)
        glfw.make_context_current(self.window)

        self.glfw_renderer = GlfwRenderer(self.window, attach_callbacks=False)
        self.controller = Controller(self.camera, self.visualizer)
        self.recorder = Recorder(self.camera, self.visualizer, initial_dir=self.get_initial_dir())

        # register callbacks
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_char_callback(self.window, self.char_callback)
        glfw.set_window_size_callback(self.window, self.resize_callback)

    def shutdown(self):
        print("done")
        self.gl_renderer.destroy_context()
        self.glfw_renderer.shutdown()
        glfw.terminate()

    def mouse_button_callback(self, window, button, action, mods):
        if not imgui.get_io().want_capture_mouse:
            self.controller.mouse_button_callback(button, action, mods)

    def cursor_pos_callback(self, window, x, y):
        self.glfw_renderer.mouse_callback(window, x, y)
        self.controller.cursor_pos_callback(x, y)

        self.last_cursor_pos = (x, y)

    def scroll_callback(self, window, xoffset, yoffset):
        self.glfw_renderer.scroll_callback(window, xoffset, yoffset)
        self.controller.scroll_callback(xoffset, yoffset)

    def key_callback(self, window, key, scancode, action, mods):
        self.glfw_renderer.keyboard_callback(window, key, scancode, action, mods)
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
        self.controller.key_callback(key, scancode, action, mods)

    def char_callback(self, window, char):
        self.glfw_renderer.char_callback(window, char)

    def resize(self, width, height):
        self.width = width
        self.height = height
        self.camera.resize(width, height)
        self.gl_renderer.resize(width, height)

    def resize_callback(self, window, width, height):
        self.glfw_renderer.resize_callback(window, width, height)
        self.resize(width, height)

    def render_callback(self, render: Render):
        last_cursor_pos = (min(self.width - 1, max(0, self.last_cursor_pos[0])),
                           min(self.height - 1, max(0, self.last_cursor_pos[1])))
        pix_id = int(last_cursor_pos[1]) * self.width + int(last_cursor_pos[0])
        world_pos = render.world_pos[pix_id].cpu().numpy() if render.world_pos is not None else None
        if world_pos is not None:
            self.controller.cursor_pos_3d_callback(world_pos[0], world_pos[1], world_pos[2])

    def start_recording(self):
        self.original_width = self.width
        self.original_height = self.height
        self.resize(self.recorder.recording_width, self.recorder.recording_height)
        self.recorder.start_recording()
        self.recording = True

    def stop_recording(self):
        self.resize(self.original_width, self.original_height)
        self.recording = False

    def draw_ui(self):
        with imgui.begin_main_menu_bar() as menu_bar:
            if menu_bar.opened:
                with imgui.begin_menu("General", True) as main_menu:
                    if main_menu.opened:
                        simple_button("Load scene", self.load_button_callback)

                        imgui.separator()

                        if self.train_cams is not None:
                            pose_changed, self.snap_pose_id = imgui.input_int("Snap to pose", self.snap_pose_id, 1, 13,
                                                                              flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                            if pose_changed:
                                self.snap_pose_id = min(max(0, self.snap_pose_id), len(self.train_cams) - 1)
                                new_pose = self.train_cams.poses(...).cpu().numpy()[self.snap_pose_id]

                                if self.snap_mode == SnapMode.CENTER_ONLY:
                                    self.camera.center = new_pose[:3, 3]
                                elif self.snap_mode == SnapMode.LOOK_AT:
                                    self.camera.look_at(*new_pose[:3, 3])
                                elif self.snap_mode == SnapMode.SNAP:
                                    self.camera.snap_to_pose(new_pose)
                                else:
                                    raise NotImplementedError(f"unknown snap mode {self.snap_mode}")

                            with imgui.begin_combo("Snap mode", self.snap_mode.value) as combo:
                                if combo.opened:
                                    for mode in SnapMode:
                                        is_selected = self.snap_mode == mode
                                        if imgui.selectable(mode.value, is_selected)[0]:
                                            self.snap_mode = mode

                                        if is_selected:
                                            imgui.set_item_default_focus()

                            imgui.separator()

                        if self.geo_reference is not None:
                            simple_button("Measure", self.measure_button_callback)

                self.recorder.draw_ui()
                self.camera.draw_ui()
                self.visualizer.draw_ui()
                self.gl_renderer.draw_ui()
                self.scene.model.draw_ui()
                self.scene.postprocessor.draw_ui()
                self.scene.background.draw_ui()

                if self.trainer is not None:
                    self.save_trainer = imgui.button("Save trainer")

                    clicked_play = imgui.button("Play" if not self.train_continuous else "Stop")
                    if clicked_play:
                        self.train_continuous = not self.train_continuous

                    imgui.text(f"Train Step: {self.trainer.global_step}")

                imgui.text(f"Current FPS: {self.current_fps:.0f}")

    def render_to_target(self, image):
        if self.recording:
            self.recorder.write_frame(image)
        else:
            self.gl_renderer.tensor_to_screen(image * 255)
            imgui.new_frame()
            self.draw_ui()
            imgui.render()
            self.glfw_renderer.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

    def render_frame(self):
        render_info = self.camera()
        gl_render = self.gl_renderer.render(render_info)
        render = self.scene(render_info.to(self.device))
        self.render_callback(render)
        image = self.visualizer(render, gl_render).reshape((self.height, self.width, 3))

        return image

    def tick(self, delta_time):
        self.controller.tick(delta_time)
        if self.recording:
            self.recorder.tick(1 / self.recorder.recording_fps)
            self.camera.tick(100)
        else:
            self.recorder.tick(delta_time)
            self.camera.tick(delta_time)
        self.gl_renderer.camera_center = self.camera.center

    @torch.no_grad()
    def render(self):
        last_frame_start_time = glfw.get_time()
        while not glfw.window_should_close(self.window):
            # region Training
            if self.trainer is not None and self.train_continuous:
                with torch.enable_grad():
                    self.trainer.train_step()
                self.gl_renderer.update_train_cams(self.train_cams)

            if trainer is not None and self.save_trainer:
                self.trainer.save(self.scene_path)
            # endregion

            frame_start_time = glfw.get_time()
            delta_time = (frame_start_time - last_frame_start_time)
            self.current_fps = 1.0 / delta_time
            last_frame_start_time = frame_start_time

            glfw.poll_events()
            self.glfw_renderer.process_inputs()

            if self.recorder.wants_to_record:
                self.start_recording()

            self.tick(delta_time)

            if self.recording and not self.recorder.playing:
                self.stop_recording()

            image = self.render_frame()

            self.render_to_target(image)


if __name__ == "__main__":
    path = None
    save_path = None
    trainer = None

    # trainer, save_path = get_trainer(debug=True)

    if trainer is not None:
        path = save_path

    gui = GUI(path, trainer=trainer, width=2448 // 3, height=2048 // 3, init_angle=74)
    gui.render()
