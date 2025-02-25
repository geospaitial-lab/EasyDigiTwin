from pathlib import Path

import geopandas as gpd

from src.preprocessing.image_converter import create_test_images

dir_path = "/path/to/data/dir"
dir_path = "/data/NAS4/Befahrung_Dormagen/"
test_output_dir_path = "/path/to/test/output/dir"
test_output_dir_path = "/data/NAS4/Befahrung_Dormagen/run_10_processed_test_2/test_images"
recording_points_path = "/path/to/recording_points.gpkg"
recording_points_path = "/data/NAS4/Befahrung_Dormagen/run_10_processed_test_2/recording_points_chemlab.gpkg"

cam_names = [
    'HMU',
    'VMU',
]

if __name__ == '__main__':
    dir_path = Path(dir_path)
    _output_dir_path = Path(test_output_dir_path)
    _output_dir_path.mkdir(exist_ok=True)

    recording_points = gpd.read_file(recording_points_path)

    create_test_images(
        recording_points=recording_points,
        cam_names=cam_names,
        dir_path=dir_path,
        output_dir_path=_output_dir_path,
        n_images=5
    )