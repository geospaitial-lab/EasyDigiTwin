import warnings

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    SegformerForSemanticSegmentation,
)

warnings.filterwarnings('ignore', message="TypedStorage is deprecated")

IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD = np.array([0.229, 0.224, 0.225])
MASK_CLASSES = np.array([10, 11, 12, 13, 14, 15, 17, 18]).astype(np.uint8)

depth_image_processor = AutoImageProcessor.from_pretrained('LiheYoung/depth-anything-large-hf',
                                                           cache_dir="data/model_cache")
depth_model = AutoModelForDepthEstimation.from_pretrained('LiheYoung/depth-anything-large-hf',
                                                          cache_dir="data/model_cache").eval().to('cuda')
segmentation_model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b5-finetuned-cityscapes-1024-1024',
    cache_dir="data/model_cache").eval().to('cuda')


@torch.no_grad()
def get_depth_map(
    image: npt.NDArray,
) -> npt.NDArray:
    inputs = depth_image_processor(images=image, return_tensors='pt').to('cuda')
    outputs = depth_model(**inputs)
    predicted_depth = outputs.predicted_depth

    interpolated_depth = torch.nn.functional.interpolate(
        predicted_depth[None, ...],
        size=image.shape[:2],
        mode='bilinear',
        align_corners=False,
    )
    interpolated_depth = interpolated_depth.squeeze().cpu().numpy()

    normalized_depth = (
        (interpolated_depth - interpolated_depth.min()) /
        (interpolated_depth.max() - interpolated_depth.min())
    )
    return (normalized_depth * (2 ** 16 - 1)).astype(np.uint16)


def _buffer_mask(
    mask: npt.NDArray,
    buffer_size: int,
) -> npt.NDArray:
    mask = torch.tensor(mask).float().unsqueeze(0)
    mask = 255 - mask
    structuring_element = torch.ones((1, 1, buffer_size, buffer_size))
    buffered_mask = F.conv2d(mask, structuring_element, padding=buffer_size // 2).clamp(0, 255)
    buffered_mask = 255 - buffered_mask
    return buffered_mask.squeeze().numpy()


def _preprocess(
    inputs: torch.Tensor,
) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        inputs,
        scale_factor=.5,
        mode='bilinear',
    )


def _postprocess(
    logits: torch.Tensor,
) -> torch.Tensor:
    logits = torch.nn.functional.interpolate(
        logits,
        scale_factor=8,
        mode='bilinear',
    )
    mask = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
    mask = np.where(np.isin(mask, MASK_CLASSES), 0, 255).astype(np.uint8)
    return mask


@torch.no_grad()
def get_mask(
    image: npt.NDArray,
    cam_name: str,
    vmu_mask: npt.NDArray,
    hmu_mask: npt.NDArray,
) -> npt.NDArray:
    image = (image / 255. - IMAGE_MEAN) / IMAGE_STD
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to('cuda')

    inputs = _preprocess(image)
    logits = segmentation_model(inputs).logits
    outputs = _postprocess(logits).squeeze(0)

    if cam_name == 'VMU':
        outputs[vmu_mask == 0] = 0
    elif cam_name == 'HMU':
        outputs[hmu_mask == 0] = 0

    return _buffer_mask(outputs, 30)
