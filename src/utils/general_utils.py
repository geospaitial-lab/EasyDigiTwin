import numpy as np
import torch


C_0 = 0.28209479177387814


def inv_sigmoid(x):
    return torch.log(x / (1 - x))


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relative_image_gradient_magnitude(image: torch.Tensor):
    bottom = image[2:, 1:-1]
    top = image[:-2, 1:-1]
    right = image[1:-1, 2:]
    left = image[1:-1, :-2]

    max_gradient = torch.maximum(torch.abs(top-bottom).mean(dim=-1), torch.abs(right-left).mean(dim=-1))
    norm_gradient = (max_gradient - max_gradient.min()) / (max_gradient.max() - max_gradient.min())
    norm_gradient = torch.nan_to_num(norm_gradient, 0.0)

    return torch.nn.functional.pad(norm_gradient, (1, 1, 1, 1), value=1.0)

def minpool_grayscale(image: torch.Tensor, kernel_size=5):
    image_shape = image.shape
    minpooled_image = -torch.nn.functional.max_pool2d(-image.reshape(1, -1, image_shape[-2], image_shape[-1]),
                                                      kernel_size=kernel_size,
                                                      padding=kernel_size//2,
                                                      stride=1).reshape(image_shape)

    return minpooled_image


def squared_normalized_cross_correlation(a, b):
    a_zero_mean = a - a.mean(-1, keepdim=True)
    b_zero_mean = b - b.mean(-1, keepdim=True)

    cross = (a_zero_mean * b_zero_mean).sum(-1)
    a_var = a_zero_mean.pow(2).sum(-1)
    b_var = b_zero_mean.pow(2).sum(-1)

    cc = cross * cross / (a_var * b_var + 1e-8)
    return cc
