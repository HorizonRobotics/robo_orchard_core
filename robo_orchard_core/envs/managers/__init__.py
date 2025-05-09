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

"""The managers to interact with the environment.

`Manager` is a design pattern that handles multiple entities of the same type
and provides a unified interface to interact with them. For example, in the
context of the environment, the manager can be responsible for certain
aspects of the environment, such as managing the actions, observations, and
events.

Managers are used to manage ManagerTerms, which are the basic blocks to
interact with the environment. Each manager term is responsible for a specific
implementation, such as observations, rewards, etc.

Manager handles all activities related to the environment, and it also
follows the `Observer` design pattern to notify the agents about the changes
in the environment and trigger the events.
"""
