from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from tqdm import tqdm

from src.utils.image_utils import (
    bayer_to_rgb,
    make_uint8,
    normalize_colors,
    undistort,
)
from src.preprocessing.predictor import get_depth_map, get_mask


def get_image(image_path):
    image = np.load(image_path)
    image = bayer_to_rgb(image)
    image = normalize_colors(image, bit_depth=8)
    image = undistort(image)
    image = make_uint8(image)

    return image


def convert_images(
    recording_points: gpd.GeoDataFrame,
    cam_names: list[str],
    dir_path: Path,
    output_dir_path: Path,
    vmu_mask_path: Path,
    hmu_mask_path: Path,
) -> None:
    images_dir_path = output_dir_path / 'images'
    depths_dir_path = output_dir_path / 'depth'
    masks_dir_path = output_dir_path / 'masks'

    images_dir_path.mkdir(exist_ok=True)
    masks_dir_path.mkdir(exist_ok=True)
    depths_dir_path.mkdir(exist_ok=True)

    vmu_mask = cv2.imread(str(vmu_mask_path))
    vmu_mask = cv2.cvtColor(vmu_mask, cv2.COLOR_BGR2GRAY)
    hmu_mask = cv2.imread(str(hmu_mask_path))
    hmu_mask = cv2.cvtColor(hmu_mask, cv2.COLOR_BGR2GRAY)

    for i, row in tqdm(recording_points.iterrows(), total=len(recording_points)):
        for j, cam_name in enumerate(cam_names):
            image_path = dir_path / row[f'{cam_name}_raw']

            image = get_image(image_path)

            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            (images_dir_path / cam_name).mkdir(exist_ok=True)
            cv2.imwrite(str(images_dir_path / row[cam_name]), bgr_image.astype(np.uint8))

            depth_map = get_depth_map(image)
            (depths_dir_path / cam_name).mkdir(exist_ok=True)
            np.save(str(depths_dir_path / (row[cam_name] + '.npy')), depth_map)

            mask = get_mask(
                image,
                cam_name=cam_name,
                vmu_mask=vmu_mask,
                hmu_mask=hmu_mask,
            )
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            (masks_dir_path / cam_name).mkdir(exist_ok=True)
            cv2.imwrite(str(masks_dir_path / (row[cam_name] + '.png')), mask.astype(np.uint8))


def create_test_images(
        recording_points: gpd.GeoDataFrame,
        cam_names: list[str],
        dir_path: Path,
        output_dir_path: Path,
        n_images = 5,
) -> None:
    output_dir_path.mkdir(exist_ok=True)

    for i, row in tqdm(recording_points.head(n_images).iterrows(), total=n_images):
        for j, cam_name in enumerate(cam_names):
            image_path = dir_path / row[f'{cam_name}_raw']

            image = get_image(image_path)

            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            (output_dir_path / cam_name).mkdir(exist_ok=True)
            cv2.imwrite(str(output_dir_path / row[cam_name]), bgr_image.astype(np.uint8))
