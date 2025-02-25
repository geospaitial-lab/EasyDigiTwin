import json
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm

from src.datasets.datasets import Dataset
from src.scenes.full_scene import FullScene
from src.training.loggers import BaseLogger
from src.training.losses import CompositeLoss, BaseLoss
from src.utils.callback_utils import CallbackList

base_config = {"num_epochs": 30,
               "steps_per_epoch": 1000,
               "batchsize": 8192,
               "device": "cpu",
               "accumulation_steps": 1,
               "use_amp": False
               }


class StatsForProgbar:
    def __init__(self):
        self.sums_dict = {}

    def __call__(self, batch, log_dict):
        print_list = []
        for k, v in log_dict.items():
            if k == "step":
                continue

            if k not in self.sums_dict or batch == 0:
                self.sums_dict[k] = 0

            if k.endswith("_"):
                self.sums_dict[k] = v
                denominator = 1
            else:
                self.sums_dict[k] += v
                denominator = batch + 1

            if k.endswith("loss"):
                print_list.append(f"{k}: {self.sums_dict[k] / denominator:.6f}")
            else:
                print_list.append(f"{k}: {self.sums_dict[k] / denominator:.2f}")

        return " ".join(print_list)


class Trainer:
    def __init__(self, scene_model: FullScene,
                 train_dataset: Dataset,
                 loss: BaseLoss | CompositeLoss,
                 optimizers: dict,
                 schedulers: dict | None = None,
                 val_dataset=None,
                 config: dict | None = None,
                 logger: BaseLogger | None = None,
                 log_frequency: int = 100):

        # read and parse config
        self.train_config = base_config.copy()
        if config is not None:
            self.train_config.update(config)

        self.device = self.train_config["device"]
        self.num_epochs = self.train_config["num_epochs"]
        self.accumulation_steps = self.train_config["accumulation_steps"]
        self.steps_per_epoch = self.train_config["steps_per_epoch"]
        self.train_dataset = train_dataset
        self.val_dataloader = None
        if val_dataset is not None:
            self.val_dataloader = DataLoader(val_dataset, num_workers=os.cpu_count() // 4, batch_size=None,
                                             pin_memory=True)

        self.scene_model = scene_model.to(self.device)

        self.loss = loss

        self.optimizers = optimizers
        self.schedulers = {}
        if schedulers is not None:
            self.schedulers = schedulers

        # register callbacks
        self.callbacks = CallbackList()
        self.callbacks.add_callbacks(self.scene_model.get_callbacks())
        self.callbacks.add_callbacks(self.loss.get_callbacks())
        self.callbacks.add_callbacks(self.train_dataset.get_callbacks())

        # set up training
        self.global_step = 0
        self.current_batch = 0
        self.use_amp = self.train_config["use_amp"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.psnr = PeakSignalNoiseRatio(data_range=1).to(self.device)
        self.logs = {"loss": [],
                     "psnr": [],
                     "step": []}

        self.logger = logger
        if self.logger is not None:
            parameters = {"train_config": self.train_config,
                          "dataset": self.train_dataset.config,
                          "losses": self.loss.config}
            self.logger.log_parameters(parameters)

        self.log_frequency = log_frequency

    def train_step(self):
        repeat = True  # there must be a better way...
        while repeat:
            repeat = False
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                self.callbacks.on_step_begin(self.global_step)

                render_infos, target_renders = self.train_dataset.get_random_batch()
                renders = []
                for target_render, render_info in zip(target_renders, render_infos):
                    target_render = target_render.to(self.device)
                    render_info = render_info.to(self.device)

                    render = self.scene_model(render_info)
                    renders.append(render)

                    if render is None:
                        repeat = True
                        continue

                    self.callbacks.on_render(render)

                if repeat:
                    continue

                loss_dict = self.loss(renders, target_renders, render_infos)

                loss = sum(_loss for _loss in loss_dict.values())

            self.scaler.scale(loss).backward()

            self.callbacks.on_backward(self.global_step, grad_scale=self.scaler.get_scale())

            if self.global_step % self.accumulation_steps == self.accumulation_steps - 1:
                self.callbacks.on_pre_optimizers_step(self.global_step, self.optimizers)
                for opt in self.optimizers.values():
                    self.scaler.step(opt)
                self.callbacks.on_post_optimizers_step(self.global_step, self.optimizers)

                self.scaler.update()

                for opt in self.optimizers.values():
                    opt.zero_grad()

            for sch in self.schedulers.values():
                sch.step()

            log_dict = None
            if self.global_step % self.log_frequency == 0:
                with torch.no_grad():
                    render_rgbs = render.features
                    if target_render.valid_mask is not None:
                        render_rgbs *= target_render.valid_mask
                    target_rgbs = target_render.rgbs
                    if target_render.valid_mask is not None:
                        target_rgbs *= target_render.valid_mask
                    psnr = self.psnr(render_rgbs, target_rgbs)
                    self.psnr.reset()

                    log_dict = {
                        "loss": loss.detach().cpu().float(),
                        "psnr": psnr.cpu().float(),
                        "step": self.global_step
                    }

                    log_dict.update(render.logs)

                    cpu_loss_dict = {k: v.cpu().float() for k, v in loss_dict.items()}

                    if self.logger is not None:
                        self.logger.log_losses(cpu_loss_dict)
                        self.logger.log_metrics(log_dict)
                        self.logger.log_optimizers()

                    log_dict.update(cpu_loss_dict)

            self.callbacks.on_step_end(self.global_step)

            self.global_step += 1
            self.current_batch += 1

            if self.current_batch >= self.steps_per_epoch:
                self.current_batch = 0
                torch.cuda.empty_cache()

            return log_dict

    def fit(self):
        stats = StatsForProgbar()
        self.callbacks.on_train_begin()
        for epoch in range(self.num_epochs):
            self.callbacks.on_epoch_begin(epoch)
            torch.cuda.empty_cache()
            progbar = tqdm(range(self.steps_per_epoch), desc=f"Epoch {epoch}: ")

            for batch in progbar:
                log_dict = self.train_step()
                if log_dict is not None:
                    progbar.set_postfix_str(stats(batch, log_dict))
            self.callbacks.on_epoch_end(epoch)
        self.callbacks.on_train_end()

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        train_config_path = os.path.join(path, "train_config.json")
        with open(train_config_path, "w") as file:
            json.dump(self.train_config, file)

        self.scene_model.save(path)
        self.train_dataset.save(path)
        self.loss.save(path)
