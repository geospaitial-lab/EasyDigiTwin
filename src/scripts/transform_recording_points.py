from pathlib import Path

import pandas as pd

from src.preprocessing.csv_parser import csv_to_gdf

dir_path = "/path/to/data/dir"
run_id = 0
output_path = "/path/to/recording_points.gpkg"
csv_path = "/path/to/gnss.csv"


if __name__ == '__main__':
    dir_path = Path(dir_path)
    csv_path = Path(csv_path)
    _output_path = Path(output_path)
    _output_path.parent.mkdir(exist_ok=True)
    df = pd.read_csv(csv_path, sep=';', decimal=',')

    recording_points = csv_to_gdf(
        dir_path=dir_path,
        run_id=run_id,
        df=df,
    )

    recording_points.to_file(str(_output_path), driver='GPKG')

