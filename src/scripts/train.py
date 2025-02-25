import os
import warnings

import sys
from pathlib import Path

from torch.optim import Adam
from neptune.common.warnings import NeptuneWarning

from src.scenes.backgrounds import ConstantBackgroundModel, GSSkyboxBackgroundModel
from src.datasets.camera_refinements import StandardCameraRefinement
from src.scenes.gaussian.gaussian_densifiers import GaussianDensifier, ScreenGradientDensifier, MCMCDensifier, \
    StreetScreenGradientDensifier
from src.scenes.gaussian.gaussian_models import GaussianModel
from src.scenes.gaussian.gaussian_renderers import GaussianRenderer
from src.training.loggers import NeptuneLogger
from src.training.losses import (CompositeLoss, GSSparsityLoss, PatchPearsonLoss, PearsonLoss, GSOpacityLoss,
                                 GSNormalLoss, GSDistortionLoss, GSScaleLoss, RGBLossL1, RGBLossSSIM,
                                 RGBLossMSSSIM, RGBLossMSE, GSFlatnessLoss, GSDepthRankingLoss)
from src.scenes.scene_model import Gaussian
from src.scenes.full_scene import FullScene
from src.training.schedulers import GSScheduler, ExpCosScheduler
from src.training.trainer import Trainer

from src.datasets.datasets import ColmapDataset

warnings.filterwarnings("ignore", message="To copy construct from a tensor")
warnings.filterwarnings("ignore", category=NeptuneWarning)


def get_trainer(debug=False):
    # region Directories

    save_dir = "/path/to/save/dir/"
    scene_name = "scene_name"
    dataset_path = "/path/to/dataset/"
    run_name = "run_name"
    neptune_project = None  # For logging in neptune

    save_path = os.path.join(save_dir, scene_name, run_name)

    # endregion

    # region General Params

    train_config = {"device": "cuda",
                    "num_epochs": 30,
                    "steps_per_epoch": 1000,
                    "mean_learning_rate": 2e-5,
                    "color_learning_rate": 2.5e-3,
                    "opacity_learning_rate": 5e-2,
                    "scale_learning_rate": 2.5e-3,
                    "rotation_learning_rate": 1e-3,
                    "refine_pose": True,
                    "pose_learning_rate": 1e-4,
                    "background_learning_rate": 1e-2,
                    "latent_learning_rate": 1e-3,
                    "accumulation_steps": 1,
                    "alpha_threshold": -1,
                    }

    # endregion

    # region Dataset
    # region Camera Refinement
    camera_refinement = None
    if train_config["refine_pose"]:
        refinement_config = {"pose_class": "SixDPoseRefinement",
                             "intrinsics_class": "IntrinsicsRefinement",
                             "egocentric": True,
                             "use_log_scale": True,
                             "use_pix_scale": True,
                             "use_principal_point": True,
                             "use_distortion": False}
        camera_refinement = StandardCameraRefinement(refinement_config)
    # endregion

    train_dataset = ColmapDataset(Path(dataset_path),
                                  downsample=0.25,
                                  camera_refinement=camera_refinement,
                                  num_workers=20,
                                  flip=True,
                                  load_mask=True,
                                  load_depth=True,
                                  scale_factor=4.0,
                                  device=train_config["device"]
                                  )

    # endregion

    # region Model

    gaussian_model = GaussianModel.from_initial_points(train_dataset.initial_points_xyz,
                                                       train_dataset.initial_points_rgb,
                                                       sh_degree=3, use_mip_filter=True,
                                                       device=train_config["device"],
                                                       compute_filter_frequency=train_config["steps_per_epoch"],
                                                       )

    gaussian_model.set_cameras(train_dataset.cameras)

    renderer = GaussianRenderer(sh_degree=gaussian_model.sh_degree, device=train_config["device"],
                                sh_update_frequency=train_config["steps_per_epoch"],
                                sh_update_start=train_config["steps_per_epoch"])


    densifier = ScreenGradientDensifier(gaussian_model, densification_end=train_config["steps_per_epoch"] * 21,
                                        opacity_reduction_frequency=train_config["steps_per_epoch"] * 3,
                                        densification_start=train_config["steps_per_epoch"] // 2,
                                        densification_frequency=train_config["steps_per_epoch"] // 10,
                                        clone_split_threshold=0.00125
                                        )

    gaussians = Gaussian(gaussian_model, renderer, densifier=densifier)
    gaussians.to(train_config["device"])

    postprocessor = gaussians.get_postprocessor().to(train_config["device"])

    # endregion

    # region Loss

    losses = []
    losses.append(RGBLossL1(weight=0.8))
    losses.append(RGBLossSSIM(weight=0.2))
    losses.append(GSNormalLoss(weight=0.0, start_step=train_config["steps_per_epoch"] * 7,
                               end_step=train_config["steps_per_epoch"] * 7, end_weight=0.05))
    losses.append(GSDistortionLoss(weight=0.0, start_step=train_config["steps_per_epoch"] * 3,
                                   end_step=train_config["steps_per_epoch"] * 3, end_weight=100))
    losses.append(GSSparsityLoss(weight=0.0,
                                 start_step=train_config["steps_per_epoch"] * 9,
                                 end_step=train_config["steps_per_epoch"] * 15,
                                 end_weight=1e-6))
    losses.append(GSScaleLoss(weight=0.0,
                              start_step=train_config["steps_per_epoch"] * 9,
                              end_step=train_config["steps_per_epoch"] * 15,
                              end_weight=1e-6))
    losses.append(GSDepthRankingLoss(weight=0.2))

    loss = CompositeLoss(losses)

    # endregion

    # region Background

    background = ConstantBackgroundModel([0, 0, 0])
    # background = GSSkyboxBackgroundModel()

    # endregion

    # region Optimizers

    optimizers = {}
    schedulers = {}

    # region Gaussians
    param_lrs = {"means": train_config["mean_learning_rate"],
                 "sh_ks_diffuse": train_config["color_learning_rate"],
                 "sh_ks_specular": train_config["color_learning_rate"] / 20,
                 "opacities": train_config["opacity_learning_rate"],
                 "scales": train_config["scale_learning_rate"],
                 "rotations": train_config["rotation_learning_rate"]}
    optimizer = Adam([{"params": params, "name": name, "lr": param_lrs[name]}
                      for name, params in gaussian_model.named_parameters()], eps=1e-15, fused=True)
    optimizers["gaussians"] = optimizer

    scheduler = GSScheduler(optimizer, train_config["mean_learning_rate"] / 10,
                            n_steps=train_config["num_epochs"] * train_config["steps_per_epoch"])
    schedulers["gaussians"] = scheduler

    # endregion
    # region Cameras
    if train_config["refine_pose"]:
        camera_optimizer = Adam(train_dataset.parameters(), train_config["pose_learning_rate"],
                                eps=1e-15, fused=True)

        camera_scheduler = ExpCosScheduler(camera_optimizer, train_config["pose_learning_rate"] / 100,
                                           n_steps=train_config["num_epochs"] * train_config["steps_per_epoch"],
                                           cos_after_n_steps=train_config["num_epochs"] *
                                                             train_config["steps_per_epoch"] // 4,
                                           n_warmup_steps=train_config["steps_per_epoch"] * 5
                                           )

        optimizers["cameras"] = camera_optimizer
        schedulers["cameras"] = camera_scheduler

    # region Background
    if background.is_trainable:
        background.to(train_config["device"])
        background_param_lrs = {"sh_ks_diffuse": train_config["color_learning_rate"],
                                "sh_ks_specular": train_config["color_learning_rate"] / 20,
                                "opacities": train_config["opacity_learning_rate"],
                                "scales": train_config["scale_learning_rate"],
                                "rotations": train_config["rotation_learning_rate"]}
        optimizer = Adam([{"params": params, "name": name, "lr": background_param_lrs[name]}
                          for name, params in background.named_parameters()
                          if name != "means" and name != "device_tracker"], eps=1e-15, fused=True)
        optimizers["background"] = optimizer
    # endregion

    # endregion

    scene_model = FullScene(gaussians, postprocessor, background)

    if (hasattr(sys, 'gettrace') and sys.gettrace() is not None) or debug:
        logger = None
        print("Not logging in debug mode!")
    else:
        if neptune_project is not None:
            logger = NeptuneLogger(project_name=neptune_project, experiment_name=f"{scene_name}/{run_name}",
                                   optimizers=optimizers)

            logger.log_value("scene_name", scene_name)
        else:
            logger = None
            print("Not logging because no neptune project is defined!")

    trainer = Trainer(scene_model=scene_model,
                      train_dataset=train_dataset,
                      loss=loss,
                      optimizers=optimizers,
                      schedulers=schedulers,
                      config=train_config,
                      logger=logger,
                      log_frequency=1)

    return trainer, save_path


if __name__ == "__main__":
    trainer, save_path = get_trainer(debug=False)

    trainer.fit()

    trainer.save(save_path)
