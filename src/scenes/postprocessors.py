import os

import torch

from src.utils.interfaces import Render, RenderInfo, RenderState
from src.utils.save_load_utils import save_to_checkpoint, load_from_checkpoint


class BasePostProcessor(torch.nn.Module):
    def __init__(self):
        super(BasePostProcessor, self).__init__()
        self.config = {k: v for k, v in vars().items() if k != "self" and k != "__class__"}
        self.config["class"] = type(self).__name__
        self.device_tracker = torch.nn.Parameter(torch.empty(0))
        self.is_trainable = False
        self.training_ = False

    def forward(self, render: Render, render_info: RenderInfo) -> Render:
        raise NotImplementedError

    def get_callbacks(self):
        return []

    def save(self, path):
        save_path = os.path.join(path, "postprocessor.ckpt")
        save_to_checkpoint(self, save_path)

    def draw_ui(self):
        pass


def get_postprocessor(classname):
    return eval(classname)


def load_postprocessor(path, device="cpu"):
    if os.path.isdir(path):
        path = os.path.join(path, "postprocessor.ckpt")

    postprocessor = load_from_checkpoint(path, get_postprocessor, device=device)

    return postprocessor


class NoOpPostProcessor(BasePostProcessor):
    def forward(self, render, render_info):
        render.state = RenderState.POSTPROCESSED
        return render
