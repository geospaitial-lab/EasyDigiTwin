import geopandas as gpd
import numpy as np
import pandas as pd


def _yaw_pitch_roll_to_rot(
    yaw: float,
    pitch: float,
    roll: float,
    degrees: bool = True,
) -> np.ndarray:
    yaw = np.deg2rad(yaw) if degrees else yaw
    pitch = np.deg2rad(pitch) if degrees else pitch
    roll = np.deg2rad(roll) if degrees else roll

    yaw = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
    )
    pitch = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
    )
    roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
    )

    return yaw @ pitch @ roll


def create_geo_reference(
    recording_points: gpd.GeoDataFrame,
    cam_names: list[str],
    cam_alignments: dict[str, dict[str, float]],
) -> pd.DataFrame:
    positions = []

    for i, row in recording_points.iterrows():
        rot_axis_transform = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
        )

        f2w = np.eye(4)
        rot = _yaw_pitch_roll_to_rot(row.yaw, row.pitch, row.roll, degrees=True)
        f2w[:3, :3] = rot_axis_transform[..., :3, :3] @ rot
        f2w[:3, 3] = np.array([row.geometry.x, row.geometry.y, row.height])

        for cam_name in cam_names:
            axis_transform = np.array(
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ],
            )
            cam_alignment = cam_alignments[cam_name]

            c2f = np.eye(4)
            c2f[:3, :3] = _yaw_pitch_roll_to_rot(
                cam_alignment['yaw'],
                cam_alignment['pitch'],
                cam_alignment['roll'],
                degrees=True,
            )
            c2f[:3, 3] = np.array([cam_alignment['x'], cam_alignment['y'], cam_alignment['z']])

            c2f = c2f @ axis_transform  # axis-transform before c2f -> all rotations and translations in frame's axes
            c2w = f2w @ c2f

            filename = row[cam_name].replace('.npy', '.png')

            position = {'filename': filename, 'X': c2w[0, 3], 'Y': c2w[1, 3], 'Z': c2w[2, 3]}
            positions.append(position)

    return pd.DataFrame(positions)
