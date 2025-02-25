from enum import Enum

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from roma import rotmat_geodesic_distance

from src.training.losses import sample_depth_loss_indices
from src.datasets.data_loaders import Loader


class SampleMode(str, Enum):
    ALL = "all"
    DEPTH = "depth"
    RAYS = "rays"


class DataSampler(Dataset):
    def __init__(self,
                 loader: Loader,
                 batch_size: int = 8192,
                 downsample: float = 1.0,
                 sample_mode: SampleMode = SampleMode.ALL,
                 seed: int = None):

        self.loader = loader

        self.width = self.loader.width
        self.height = self.loader.height

        self.batch_size = batch_size

        self.downsample = downsample
        self.sample_mode = sample_mode

        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, image_idx) -> tuple[dict[str, torch.Tensor | None], int, np.ndarray]:

        rgbs, valid_rgbs_mask, depths = self.loader.load(image_idx)

        if self.sample_mode == SampleMode.ALL:
            ray_idx = ...
        elif self.sample_mode == SampleMode.DEPTH:
            assert depths is not None
            assert valid_rgbs_mask is not None
            ray_idx = sample_depth_loss_indices(depths,
                                                self.width, self.height,
                                                valid_rgbs_mask,
                                                self.batch_size // 4,
                                                rank_patch_size=300,
                                                knn_crop_radius=3)
        elif self.sample_mode == SampleMode.RAYS:
            ray_idx = self.rng.choice(self.width * self.height, self.batch_size)
        else:
            raise NotImplementedError

        rgbs = rgbs[ray_idx]
        valid_rgbs_mask = valid_rgbs_mask[ray_idx] if valid_rgbs_mask is not None else []
        depths = depths[ray_idx] if depths is not None else []

        target_dict = {"rgbs": rgbs, "valid_rgbs_mask": valid_rgbs_mask, "depths": depths}

        if ray_idx is ...:
            ray_idx = -1

        return target_dict, image_idx, ray_idx


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.parent_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.parent_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.parent_iterator = super().__iter__()
            batch = next(self.parent_iterator)
        return batch
