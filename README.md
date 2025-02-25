# Installation

## Pre-requisites
- Make sure [CUDA](https://developer.nvidia.com/cuda) is installed
- Install [pytorch](https://pytorch.org/get-started/locally/) with support for the installed CUDA version

## Requirements
```bash
pip install -r requirements.txt
```

## Docker (Linux only)
Alternatively you can use [Docker](https://www.docker.com/) to run the code.
Build an image with the provided [Dockerfile](Dockerfile):
```bash
docker build -t easy_digi_twin .
```
Make sure that [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) is installed
and used as the default runtime.

# Preprocessing
## Transform Recording Points
Modify [transform_recording_points.py](src/scripts/transform_recording_points.py) to set the following parameters:
```python
dir_path = "/path/to/data/dir"
run_id = 0
output_path = "/path/to/recording_points.gpkg"
csv_path = "/path/to/gnss.csv"
```

## Create Test Images
Modify [create_test_images.py](src/scripts/create_test_images.py) to set the following parameters:
```python
dir_path = "/path/to/data/dir"
output_dir_path = "/path/to/test/output/dir"
recording_points_path = "/path/to/recording_points.gpkg"
```

## Preprocess Data
Modify [preprocess.py](src/scripts/preprocess.py) to at least set the following parameters:

```python
dir_path = "/path/to/data/dir"
output_dir_path = "/path/to/output/dir"
recording_points_path = "/path/to/recording_points.gpkg"
vmu_mask_path = "/path/to/VMU_Mask.png"
hmu_mask_path = "/path/to/HMU_Mask.png"
```

# Training
Modify [train.py](src/scripts/train.py) to at least set the following parameters:

```python
save_dir = "/path/to/save/dir/"
scene_name = "scene_name"
dataset_path = "/path/to/dataset/"
run_name = "run_name"
```

The rest of the file should be modified to adjust the optimization process.

# Georeference Scene
Modify [georeference_model.py](src/scripts/georeference_model.py) to set the following parameters:

```python
scene_path = "/path/to/saved/scene"
reference_file_path = "/path/to/geo_reference.txt"
```

# GUI

1. Run [gui.py](src/gui/gui.py) 
2. Select General -> Load Scene from the menu
3. Select .ply or scene_model.json to load

# LICENSE
All code in this repository is licensed under the [GPL-3.0 License](LICENSE).
Installed requirements are listed in [requirements.txt](requirements.txt) and 
are licensed under their respective licenses.

**NOTE**: The installed dependency [EasyDigiTwin-gaussian-rasterization](https://github.com/geospaitial-lab/EasyDigiTwin-gaussian-rasterization.git)
is licensed under the [Gaussian-Splatting License](https://github.com/geospaitial-lab/EasyDigiTwin-gaussian-rasterization/blob/main/LICENSE.md)
which prohibits commercial use.
The SegFormer Model used for preprocessing is licensed under 
the [NVIDIA Source Code License for SegFormer](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)
which also prohibits commercial use.
Therefore, the code in this repository can not be run for commercial use without replacing these dependencies.