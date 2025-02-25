from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd


CAM_NAMES = [
    'HLO',
    'HLU',
    'HMO',
    'HMU',
    'HRO',
    'HRU',
    'O',
    'VLO',
    'VLU',
    'VMO',
    'VMU',
    'VRO',
    'VRU',
]
CENTRAL_MERIDIAN = 9
DISK_1_NAME = 'disk1'
DISK_2_NAME = 'disk2'
EPSG_CODE = 25832


def _grid_convergence(
    long: npt.NDArray,
    lat: npt.NDArray,
) -> npt.NDArray:
    long_rad = np.deg2rad(long)
    lat_rad = np.deg2rad(lat)
    central_meridian_rad = np.deg2rad(CENTRAL_MERIDIAN)
    return np.arctan(np.tan(long_rad - central_meridian_rad) * np.sin(lat_rad)) / np.pi * 180


def _gather_image_paths(
    disk_dir_path: Path,
    run_id: int,
) -> list[Path]:
    dir_path = disk_dir_path.parents[0]
    return [
        image_path.relative_to(dir_path)
        for image_path in disk_dir_path.iterdir()
        if image_path.suffix == '.npy' and f'run_{run_id}' in image_path.stem
    ]


def _parse_image_paths(
    image_paths: list[Path],
) -> list[_ImageMetadata]:
    images_metadata = [
        _parse_image_path(image_path)
        for image_path in image_paths
    ]
    return sorted(
        images_metadata,
        key=lambda image_metadata: image_metadata.image_id,
    )


def _parse_image_path(
    image_path: Path,
) -> _ImageMetadata:
    split_image_stem = image_path.stem.split('_')
    split_image_stem = [item for item in split_image_stem if item]
    _, run_id, cam_name, _, image_id, _, time_stamp = split_image_stem
    processed_path = Path(cam_name) / f'frame_{image_id}.png'
    return _ImageMetadata(
        run_id=int(run_id),
        cam_name=cam_name,
        image_id=int(image_id),
        time_stamp=int(time_stamp),
        image_path=str(image_path),
        processed_path=str(processed_path),
    )


def _index_df(
    df: pd.DataFrame,
    images_metadata: list[_ImageMetadata],
) -> pd.DataFrame:
    start_time_stamp = images_metadata[0].time_stamp
    start_index = abs(df['Time (GPS ns)'] - start_time_stamp).argmin()

    assert len(images_metadata) % len(CAM_NAMES) == 0
    num_images = len(images_metadata) // len(CAM_NAMES)
    assert images_metadata[-1].image_id == num_images - 1

    return df[start_index:start_index + num_images]


def _build_gdf(
    images_metadata: list[_ImageMetadata],
    df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    recording_points = defaultdict(list)

    for image_metadata in images_metadata:
        cam_name = image_metadata.cam_name
        recording_points[cam_name].append(image_metadata.processed_path)
        recording_points[f'{cam_name}_raw'].append(image_metadata.image_path)
        recording_points['image_id'].append(image_metadata.image_id)

    num_images = len(images_metadata) // len(CAM_NAMES)

    recording_points.update({
        'datetime': df['Time (yyyy-MM-dd HH:mm:ss.fff)'],
        'time_gps_ns': df['Time (GPS ns)'],
        'height': df['Altitude (m)'],
        'vel_north': df['Velocity north (m/s)'],
        'vel_east': df['Velocity east (m/s)'],
        'vel_down': df['Velocity down (m/s)'],
        'accel_x': df['Acceleration Xv (m/s²)'],
        'accel_y': df['Acceleration Yv (m/s²)'],
        'accel_z': df['Acceleration Zv (m/s²)'],
        'yaw': df['Heading (deg)'] - _grid_convergence(df['Longitude (deg)'].values, df['Latitude (deg)'].values),
        'pitch': df['Pitch (deg)'],
        'roll': df['Roll (deg)'],
        'acc_north': df['Position accuracy north (m)'],
        'acc_east': df['Position accuracy east (m)'],
        'acc_down': df['Position accuracy down (m)'],
        'image_id': np.arange(num_images),
    })

    gdf = gpd.GeoDataFrame(
        recording_points,
        geometry=gpd.points_from_xy(df['Longitude (deg)'], df['Latitude (deg)']),
        crs='EPSG:4258',
    )
    gdf = gdf.to_crs(f'EPSG:{EPSG_CODE}')
    return gdf.sort_index(axis=1)


@dataclass
class _ImageMetadata:
    run_id: int
    cam_name: str
    image_id: int
    time_stamp: int
    image_path: str
    processed_path: str


def csv_to_gdf(
    dir_path: Path,
    run_id: int,
    df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    disk_1_path = dir_path / DISK_1_NAME
    disk_2_path = dir_path / DISK_2_NAME

    with ThreadPoolExecutor() as executor:
        tasks = list(executor.map(_gather_image_paths, [disk_1_path, disk_2_path], [run_id, run_id]))

    image_paths = tasks[0] + tasks[1]
    images_metadata = _parse_image_paths(image_paths)
    df = _index_df(df, images_metadata)
    gdf = _build_gdf(images_metadata, df)
    columns_1 = [name for cam in CAM_NAMES for name in (cam, f'{cam}_raw')]
    columns_2 = [column for column in gdf.columns if column not in columns_1]
    gdf = gdf[columns_2 + columns_1]
    return gdf
