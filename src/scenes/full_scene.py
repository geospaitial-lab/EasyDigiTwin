from src.scenes.backgrounds import BaseBackgroundModel, load_background
from src.utils.interfaces import RenderInfo
from src.scenes.postprocessors import BasePostProcessor, load_postprocessor
from src.scenes.scene_model import BaseScene, load_scene_model


class FullScene:
    def __init__(self, model: BaseScene, postprocessor: BasePostProcessor, background: BaseBackgroundModel):
        self.model = model
        self.postprocessor = postprocessor
        self.background = background

    def __call__(self, render_info: RenderInfo):
        render = self.model(render_info)
        if render is None:
            return None

        render = self.postprocessor(render, render_info)
        render = self.background.composit(render, render_info)

        return render

    def to(self, device):
        self.model = self.model.to(device)
        self.postprocessor = self.postprocessor.to(device)
        self.background = self.background.to(device)

        return self

    def get_callbacks(self):
        callback_list = (self.model.get_callbacks()
                         + self.postprocessor.get_callbacks()
                         + self.background.get_callbacks())

        return callback_list

    def save(self, path):
        self.model.save(path)
        self.postprocessor.save(path)
        self.background.save(path)

    @classmethod
    def load(cls, path, device="cpu"):
        model = load_scene_model(path, device=device)
        postprocessor = load_postprocessor(path, device=device)
        background = load_background(path, device=device)

        return cls(model, postprocessor, background)
