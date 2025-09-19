# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import Any

import ray
import torch
from ray.util.state import get_actor as get_actor_state
from typing_extensions import Generic, TypeVar

from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType,
    Config,
    ConfigInstanceOf,
)
from robo_orchard_core.utils.logging import LoggerManager

logger = LoggerManager().get_child(__name__)


__all__ = [
    "DEFAULT_RAY_INIT_CONFIG",
    "is_ray_actor_alive",
    "RayActorNotAliveError",
    "RayActorDiedError",
    "RayRemoteClassConfig",
    "RayRemoteInstanceConfig",
    "RayRemoteInstance",
]

DEFAULT_RAY_INIT_CONFIG = {
    "address": None,
    "num_cpus": None,
    "num_gpus": None,
    "resources": None,
    "object_store_memory": None,
    "local_mode": False,
    "ignore_reinit_error": False,
    "include_dashboard": None,
    "dashboard_port": None,
    "job_config": None,
    "configure_logging": True,
    "logging_level": "info",
    "logging_format": None,
    "log_to_driver": True,
    "namespace": None,
    "runtime_env": None,
}


class RayActorDiedError(Exception):
    """Exception raised when a Ray actor has died unexpectedly."""

    pass


class RayActorNotAliveError(Exception):
    """Exception raised when a Ray actor is not alive."""

    pass


def is_ray_actor_alive(
    remote, timeout: int = 10, error_info: dict | None = None
) -> bool:
    """Check if a Ray actor is alive within a timeout period.

    If the actor is died, it raises a RayActorDiedError.
    """

    def check_alive():
        actor_id = remote._actor_id.hex()
        try:
            state = get_actor_state(actor_id)
            # handle case when the task is not submitted yet
            if state is None:
                if error_info is not None:
                    error_info["error"] = f"Actor id {actor_id} not found!"
                return False
            actor_live_state: str = state.state  # type: ignore
            # raise died error if the actor is dead
            if actor_live_state == "DEAD":
                death_cause = state.death_cause[  # type: ignore
                    "creation_task_failure_context"
                ]["formatted_exception_string"]
                err_msg = f"Ray actor is dead. cause: {death_cause}"
                if error_info is not None:
                    error_info["error"] = err_msg
                raise RayActorDiedError(err_msg)
            if error_info is not None:
                error_info["error"] = f"State: {actor_live_state}"
            # only return true if the actor is alive
            return state is not None and actor_live_state == "ALIVE"
        except RayActorDiedError as e:
            raise e
        except Exception as e:
            if error_info is not None:
                error_info["error"] = str(e)
            return False

    import time

    current_time = time.time()
    while time.time() - current_time < timeout:
        if check_alive():
            return True
        time.sleep(0.1)

    return False


class RayRemoteClassConfig(Config):
    __exclude_config_type__: bool = True

    num_cpus: float | int | None = 0.5
    """Number of CPU cores to allocate to the remote actor."""
    num_gpus: float | int | None = 0.1 if torch.cuda.is_available() else None
    """Number of GPUs to allocate to the remote actor."""
    memory: int = 1 * 1024**3
    """heap memory request in bytes. Default to 1GB."""
    runtime_env: dict[str, Any] | None = None
    """The runtime environment for the remote actor.

    See https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#concepts
    """


T = TypeVar("T")

ClassConfigType_co = TypeVar(
    "ClassConfigType_co", bound=ClassConfig, covariant=True
)


class RayRemoteInstanceConfig(ClassConfig[T], Generic[T, ClassConfigType_co]):
    """Configuration for creating a Ray remote instance.

    Template parameters:
        T: The type of the remote instance.
        ClassConfigType_co: The type of the configuration for the class to be
            instantiated remotely.

    """

    class_type: ClassType[T]

    remote_class_config: RayRemoteClassConfig = RayRemoteClassConfig()
    """The configuration for the remote class."""

    ray_init_config: dict[str, Any] | None = None
    """The configuration for initializing Ray. If None, use default."""

    check_init_timeout: int = 60
    """Timeout in seconds for checking if the remote actor is initialized."""

    instance_config: ConfigInstanceOf[ClassConfigType_co]


class RayRemoteInstance(Generic[T]):
    """A class that manages a Ray remote actor instance."""

    cfg: RayRemoteInstanceConfig["RayRemoteInstance", ClassConfig[T]]

    remote_cls: Any
    """The Ray remote class."""

    _remote: Any
    """The Ray remote actor instance."""

    def __init__(self, cfg: RayRemoteInstanceConfig, **kwargs):
        self.cfg = cfg
        if not ray.is_initialized():
            if self.cfg.ray_init_config is not None:
                ray.init(**self.cfg.ray_init_config)
            else:
                ray.init(**DEFAULT_RAY_INIT_CONFIG)

        remote_cls = ray.remote(**self.cfg.remote_class_config.model_dump())(
            self.cfg.instance_config.class_type
        )
        self.remote_cls = remote_cls

        remote = remote_cls.remote(self.cfg.instance_config, **kwargs)  # type: ignore
        ray_error_info = {}
        if not is_ray_actor_alive(
            remote,
            timeout=self.cfg.check_init_timeout,
            error_info=ray_error_info,
        ):
            remote = None
            raise RayActorNotAliveError(
                f"Ray actor failed to be alive within "
                f"{self.cfg.check_init_timeout} seconds. "
                f"Reason: {ray_error_info}"
                "Please check the ray remote class config and cluster that "
                "enough resources "
                f"are available: {ray.available_resources()}"
            )
        self._remote = remote

    @property
    def remote(self) -> Any:
        """Get the Ray remote actor instance.

        Raises:
            RayActorNotAliveError: If the Ray actor is not alive.
        """
        return self._remote
