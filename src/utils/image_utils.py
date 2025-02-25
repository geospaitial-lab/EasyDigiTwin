import cv2
import numpy as np
import torch
from numpy.typing import NDArray, ArrayLike


CAM_INTRINSICS = np.array(
    [
        [1800, 0, 1218.686472310209],
        [0, 1800, 1000],
        [0, 0, 1],
    ],
)
DIST_PARAMS = np.array([-0.1621592036085751, 0.06447716349026768, 0, 0])


def bayer_to_rgb(image: NDArray) -> NDArray:
    return cv2.cvtColor(image, cv2.COLOR_BAYER_RGGB2RGB)


def normalize_colors(image: NDArray, bit_depth: int) -> NDArray:
    return image.astype(np.float64) / (2 ** bit_depth - 1)


def undistort(image: NDArray, cam_intrinsics: NDArray = CAM_INTRINSICS, dist_params: NDArray = DIST_PARAMS) -> NDArray:
    return cv2.undistort(image, cam_intrinsics, dist_params)


def _make_valid(image: NDArray) -> NDArray:
    return np.clip(np.nan_to_num(image), 0, 1)


def make_uint8(image: NDArray) -> NDArray:
    return (_make_valid(image) * 255).astype(np.uint8)


def resize_image(image: NDArray, width: int, height: int):
    return cv2.resize(image, (width, height))


def adjust_exposure(image: NDArray, factor):
    return image * factor


def apply_gamma_correction(image: NDArray | torch.Tensor, gamma=2.2):
    return image ** (1 / gamma)


def white_balance_torch(image: torch.Tensor, balance_ratios: ArrayLike):
    image = image * torch.tensor(balance_ratios, device=image.device)

    return image