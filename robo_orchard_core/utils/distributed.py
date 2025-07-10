# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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

import functools
from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist
from torch.utils.data import get_worker_info

__all__ = [
    "rank_zero_only",
    "is_dist_initialized",
    "DistInfo",
    "get_dist_info",
    "DataLoaderWorkerInfo",
    "get_dataloader_worker_info",
]


def rank_zero_only(fn):
    """Ensures a function is executed only on rank 0 in a distributed setting.

    Args:
        fn (Callable): The function to wrap.

    Returns:
        Callable: The wrapped function.
    """

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


@dataclass
class DistInfo:
    """Represents distributed information.

    Attributes:
        world_size (int): The total number of processes in the
            distributed group.
        rank (int): The rank of the current process.
    """

    world_size: int
    rank: int


def is_dist_initialized():
    """Checks if PyTorch distributed mode is initialized.

    Returns:
        bool: True if distributed mode is initialized, otherwise False.
    """
    if dist.is_available():
        return dist.is_initialized()
    else:
        return False


def get_dist_info(
    process_group: Optional[dist.ProcessGroup] = None,
) -> DistInfo:
    """Gets distributed information such as world size and rank.

    Args:
        process_group (Optional[dist.ProcessGroup]): The process group to
            query. Defaults to the global process group.

    Returns:
        DistInfo: Distributed information containing world size and rank.
    """
    if is_dist_initialized():
        rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)
    else:
        rank = 0
        world_size = 1
    return DistInfo(world_size=world_size, rank=rank)


@dataclass
class DataLoaderWorkerInfo:
    """Represents worker information in a DataLoader.

    Attributes:
        world_size (int): The total number of workers in the DataLoader.
        rank (int): The index of the current worker.
    """

    world_size: int
    rank: int


def get_dataloader_worker_info() -> DataLoaderWorkerInfo:
    """Gets information about the current DataLoader worker.

    Returns:
        DataLoaderWorkerInfo: Information about the DataLoader worker,
            including world size and rank.
    """
    worker_info = get_worker_info()
    if worker_info is not None:
        worker_idx = worker_info.id
        num_workers = worker_info.num_workers
    else:
        num_workers = 1
        worker_idx = 0
    return DataLoaderWorkerInfo(world_size=num_workers, rank=worker_idx)
