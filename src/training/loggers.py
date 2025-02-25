import os

from neptune import init_run
from dotenv import load_dotenv
from neptune.utils import stringify_unsupported

load_dotenv()

class BaseLogger:

    def log_value(self, key, value):
        raise NotImplementedError

    def log_parameters(self, parameters):
        raise NotImplementedError

    def log_series(self, key, value):
        raise NotImplementedError

    def log_losses(self, losses: dict):
        raise NotImplementedError

    def log_metrics(self, metrics: dict):
        raise NotImplementedError

    def log_optimizers(self: dict):
        raise NotImplementedError


class NeptuneLogger(BaseLogger):

    def __init__(self, project_name, experiment_name, optimizers=None):
        api_token = os.getenv("NEPTUNE_API_TOKEN")
        if api_token is None:
            raise ValueError("NEPTUNE_API_TOKEN is not set. Use .env file or set environment variable.")
        self.run = init_run(project=project_name,
                            api_token=api_token,
                            name=experiment_name)
        self.optimizers = optimizers

    def log_value(self, key, value):
        self.run[key] = value

    def log_parameters(self, parameters):
        self.run["parameters"] = stringify_unsupported(parameters)

    def log_series(self, key, value):
        self.run[key].append(value)

    def log_losses(self, losses):
        for k, v in losses.items():
            self.log_series(f"losses/{k}", v)

    def log_metrics(self, metrics):
        for k, v in metrics.items():
            self.log_series(f"metrics/{k}", v)

    def log_optimizers(self):
        if self.optimizers is not None:
            # log learning rates
            for k, v in self.optimizers.items():
                for i, param_group in enumerate(v.param_groups):
                    name = i
                    if "name" in param_group:
                        name = param_group["name"]
                    self.log_series(f"optimizers/{k}/{name}", param_group["lr"])
