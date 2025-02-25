from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch

from src.utils.image_utils import normalize_colors, resize_image


class LoadingStrategy(ABC):
    def __init__(
            self,
            width: int,
            height: int,
            paths: list[Path],
    ) -> None:
        self.width = width
        self.height = height
        self.paths = paths

    @abstractmethod
    def _load(
            self,
            idx: int,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _postprocess(
            self,
            image: np.ndarray,
            downsample: float = 1.0
    ) -> torch.Tensor:
        pass

    def load(
            self,
            idx: int,
            downsample: float = 1.0
    ) -> torch.Tensor:
        image = self._load(idx)

        return self._postprocess(image, downsample)


class Loader:
    def __init__(
            self,
            rgb_strategy: LoadingStrategy,
            mask_strategy: LoadingStrategy = None,
            depth_strategy: LoadingStrategy = None,
    ) -> None:
        self.rgb_strategy = rgb_strategy
        self.mask_strategy = mask_strategy
        self.depth_strategy = depth_strategy

        self.width = rgb_strategy.width
        self.height = rgb_strategy.height

    def load(
            self,
            idx: int,
            downsample: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        rgb = self.rgb_strategy.load(idx, downsample)
        mask = self.mask_strategy.load(idx, downsample) if self.mask_strategy is not None else None
        depth = self.depth_strategy.load(idx, downsample) if self.depth_strategy is not None else None
        return rgb, mask, depth

    def __len__(self):
        return len(self.rgb_strategy.paths)


class ConvertedRGBStrategy(LoadingStrategy):

    def _load(
            self,
            idx: int,
    ) -> np.ndarray:
        bgr = cv2.imread(str(self.paths[idx]), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        return rgb

    def _postprocess(
            self,
            image: np.ndarray,
            downsample: float = 1.0
    ) -> torch.Tensor:
        image = normalize_colors(image, 8)
        image = resize_image(image, int(self.width * downsample), int(self.height * downsample))

        return torch.tensor(image, dtype=torch.float32).reshape(-1, 3)


class BinaryMaskStrategy(LoadingStrategy):

    def _load(
            self,
            idx: int,
    ) -> np.ndarray:
        mask = cv2.imread(str(self.paths[idx]), cv2.IMREAD_UNCHANGED)[..., :3]

        return mask

    def _postprocess(
            self,
            image: np.ndarray,
            downsample: float = 1.0
    ) -> torch.Tensor:
        image = cv2.resize(image, (int(self.width * downsample), int(self.height * downsample)), cv2.INTER_NEAREST)

        return torch.tensor(image, dtype=torch.bool).reshape(-1, 3)


class DepthStrategy(LoadingStrategy):

    def _load(
            self,
            idx: int,
            downsample: float = 1.0,
    ) -> np.ndarray:
        depth = np.load(str(self.paths[idx]))

        return depth

    def _postprocess(
            self,
            image: np.ndarray,
            downsample: float = 1.0
    ) -> torch.Tensor:
        image = normalize_colors(image, 16)
        image = resize_image(image, int(self.width * downsample), int(self.height * downsample))

        return torch.tensor(image, dtype=torch.float32).reshape(-1)
