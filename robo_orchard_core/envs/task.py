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

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from robo_orchard_core.utils.config import (
    ClassInitFromConfigMixin,
)


@dataclass
class TaskEvalReturn:
    """The return type for task evaluation."""

    success_flag: torch.Tensor
    """Whether the task was successfully completed.

    If success_flag is 1, the task is considered successful.
    If success_flag is -1, the task is considered failed.
    The value of 0 indicates all other cases, such as ongoing tasks or
    tasks that are not yet evaluated.

    """
    info: dict[str, Any] | None = None
    """Additional information about the task evaluation."""


class TaskMixin(
    ClassInitFromConfigMixin,
    metaclass=ABCMeta,
):
    """Task interface.

    A task is a specific environment that has a defined goal or objective,
    and it typically provides a method to evaluate the execution of an action
    in the context of that task, for example, by returning a reward or
    determining if the task is complete/success.

    If you want to create an environment that support task interface,
    you should inherit from this class and implement the abstract methods
    and properties defined in this class.

    """

    @abstractmethod
    def evaluate(self) -> TaskEvalReturn:
        raise NotImplementedError("The `evaluate` method must be implemented.")

    @property
    @abstractmethod
    def description(self) -> str:
        """A description of the task.

        This property should return a string that describes the task,
        including its objectives, constraints, and any other relevant
        information.

        Returns:
            str: A description of the task.

        """
        raise NotImplementedError(
            "The `task_description` property must be implemented."
        )

    @property
    @abstractmethod
    def goal_condition(self) -> str:
        """The goal condition of the task.

        This property should return a string that describes the goal
        condition of the task, which is the desired outcome or state that
        the task aims to achieve.

        Returns:
            str: The goal condition of the task.

        """
        raise NotImplementedError(
            "The `goal_condition` property must be implemented."
        )

    @property
    def instructions(self) -> str:
        """Instructions for the task.

        This property should return a string that provides instructions
        on how to perform the task, including any specific steps or
        guidelines that need to be followed.

        By default, it returns the task description. User can override
        this property to provide more specific instructions if needed.

        Returns:
            str: Instructions for the task.

        """
        return self.description
