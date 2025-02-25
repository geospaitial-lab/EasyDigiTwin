import json
from pathlib import Path

import geopandas as gpd
import numpy as np

from src.preprocessing.geo_referencer import create_geo_reference
from src.preprocessing.image_converter import convert_images
from src.preprocessing.scene_graph_builder import build_scene_graph

dir_path = "/path/to/data/dir"
output_dir_path = "/path/to/output/dir"
recording_points_path = "/path/to/recording_points.gpkg"
vmu_mask_path = "/path/to/VMU_Mask.png"
hmu_mask_path = "/path/to/HMU_Mask.png"

cam_names = [
    'HLO',
    'HLU',
    'HMO',
    'HMU',
    'HRO',
    'HRU',
    'VLO',
    'VLU',
    'VMO',
    'VMU',
    'VRO',
    'VRU',
]
far_distance = 30.
view_angle = 60.
cam_alignments_path = "data/alignment.json"


if __name__ == '__main__':
    dir_path = Path(dir_path)
    _output_dir_path = Path(output_dir_path)
    _output_dir_path.mkdir(exist_ok=True)
    cam_alignments_path = Path(cam_alignments_path)
    view_angle = np.deg2rad(view_angle)
    vmu_mask_path = Path(vmu_mask_path)
    hmu_mask_path = Path(hmu_mask_path)

    with cam_alignments_path.open('r') as file:
        cam_alignments = json.load(file)

    recording_points = gpd.read_file(recording_points_path)

    geo_reference = create_geo_reference(
        recording_points=recording_points,
        cam_names=cam_names,
        cam_alignments=cam_alignments,
    )

    geo_reference.to_csv(str(_output_dir_path / 'geo_reference.txt'), sep=' ', index=False, header=False)

    scene_graph = build_scene_graph(
        recording_points=recording_points,
        cam_alignments=cam_alignments,
        far_distance=far_distance,
        view_angle=view_angle,
        cam_names=cam_names,
        angle_threshold=view_angle,
    )

    scene_graph[['path_1', 'path_2']].to_csv(str(_output_dir_path / 'pairs.txt'), sep=' ', index=False, header=False)

    convert_images(
        recording_points=recording_points,
        cam_names=cam_names,
        dir_path=dir_path,
        output_dir_path=_output_dir_path,
        vmu_mask_path=vmu_mask_path,
        hmu_mask_path=hmu_mask_path,
    )
