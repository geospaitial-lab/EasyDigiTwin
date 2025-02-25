import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import (
    Point,
    Polygon,
)


def _yaw_to_rot(
    yaw: float,
    degrees: bool = True,
) -> np.ndarray:
    yaw = np.deg2rad(yaw) if degrees else yaw

    yaw = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
    )

    return yaw


def _make_cam_df(
    recording_points: gpd.GeoDataFrame,
    cam_names: list[str],
    cam_alignments: dict[str, dict[str, float]],
) -> gpd.GeoDataFrame:
    offsets = np.array(
        [
            [
                cam_alignments[cam_name]['y'],
                cam_alignments[cam_name]['x'],
                np.deg2rad(cam_alignments[cam_name]['yaw'])
            ]
            for cam_name in cam_names
        ],
    )

    recording_points_np = np.array(
        [
            [
                row[1].geometry.x,
                row[1].geometry.y,
                np.deg2rad(row[1]['yaw'])
            ]
            for row in recording_points.iterrows()
        ],
    )

    recording_point_rotations = np.array(
        [
            _yaw_to_rot(-row[1]['yaw'])
            for row in recording_points.iterrows()
        ],
    )

    recording_points_paths = [
        [
            row[1][cam_name].replace('.npy', '.png')
            for cam_name in cam_names
        ]
        for row in recording_points.iterrows()
    ]
    recording_points_paths = np.array(recording_points_paths).reshape(-1)

    cam_names_ = [
        [cam_name]
        for cam_name in cam_names
        for _ in recording_points.iterrows()
    ]
    cam_names_ = np.array(cam_names_).reshape(-1)

    tiled_offsets = np.tile(offsets, (recording_points_np.shape[0], 1))
    repeated_recording_points_np = recording_points_np.repeat(len(cam_names), axis=0)
    repeated_recording_point_rotations = recording_point_rotations.repeat(len(cam_names), axis=0)

    cam_positions = (
            repeated_recording_points_np +
            np.einsum('ijk, ik -> ij', repeated_recording_point_rotations, tiled_offsets)
    )

    cam_df = pd.DataFrame(
        {
            'path': recording_points_paths,
            'cam_name': cam_names_,
            'yaw': cam_positions[:, 2],
            'x': cam_positions[:, 0],
            'y': cam_positions[:, 1],
        },
    )
    cam_gdf = gpd.GeoDataFrame(
        cam_df,
        geometry=gpd.points_from_xy(cam_df.x, cam_df.y),
    )
    return cam_gdf.drop(['x', 'y'], axis=1)


def _make_frustum(
    point: Point,
    orientation: float,
    far_distance: float,
    view_angle: float,
    cam_name: str = '',
) -> Polygon:
    point_2 = Point(
        point.x + far_distance * np.sin(orientation - view_angle / 2),
        point.y + far_distance * np.cos(orientation - view_angle / 2),
    )
    point_3 = Point(
        point.x + far_distance * np.sin(orientation + view_angle / 2),
        point.y + far_distance * np.cos(orientation + view_angle / 2),
    )

    if cam_name == 'O':
        point_4 = Point(
            point.x - far_distance * np.sin(orientation - view_angle / 2),
            point.y - far_distance * np.cos(orientation - view_angle / 2),
        )
        point_5 = Point(
            point.x - far_distance * np.sin(orientation + view_angle / 2),
            point.y - far_distance * np.cos(orientation + view_angle / 2),
        )
        return Polygon([point_2, point_3, point_4, point_5])
    else:
        return Polygon([point, point_2, point_3])


def _get_pairs(
    frustums: gpd.GeoDataFrame,
    angle_threshold: float,
) -> gpd.GeoDataFrame:
    intersecting_polygons = gpd.overlay(frustums, frustums, how='intersection')
    intersecting_polygons = intersecting_polygons[intersecting_polygons['path_1'] != intersecting_polygons['path_2']]

    intersecting_polygons['yaw_diff'] = np.abs(
        np.mod(intersecting_polygons['yaw_1'] - intersecting_polygons['yaw_2'] + np.pi, 2 * np.pi) - np.pi
    )

    intersecting_polygons = intersecting_polygons[intersecting_polygons['yaw_diff'] < angle_threshold]
    return intersecting_polygons


def build_scene_graph(
    recording_points: gpd.GeoDataFrame,
    cam_alignments: dict[str, dict[str, float]],
    far_distance: float,
    view_angle: float,
    cam_names: list[str],
    angle_threshold: float,
) -> gpd.GeoDataFrame:
    cam_gdf = _make_cam_df(
        recording_points=recording_points,
        cam_names=cam_names,
        cam_alignments=cam_alignments,
    )
    cam_gdf['geometry'] = cam_gdf.apply(
        lambda x: _make_frustum(x.geometry, x['yaw'], far_distance, view_angle, x['cam_name']),
        axis=1,
    )

    return _get_pairs(
        frustums=cam_gdf,
        angle_threshold=angle_threshold,
    )
