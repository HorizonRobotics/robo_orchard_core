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
from dataclasses import dataclass


@dataclass
class TaskInfo:
    """The information of a task.

    This class is used to store the information of a task, including its
    description, goal condition, and instructions.

    """

    description: str
    """A description of the task.

    This property should return a string that describes the task,
    including its objectives, constraints, and any other relevant
    information.
    """

    goal_condition: str
    """The goal condition of the task.

    This property should return a string that describes the goal
    condition of the task, which is the desired outcome or state that
    the task aims to achieve.
    """

    instructions: str | None
    """Instructions for the task.

    This property should return a string that provides instructions
    on how to perform the task, including any specific steps or
    guidelines that need to be followed.

    This property  provides more specific instructions if needed.
    """
