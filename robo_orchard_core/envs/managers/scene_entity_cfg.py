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

"""Configuration for the scene entity."""

from __future__ import annotations
from abc import abstractmethod
from typing import Any

from robo_orchard_core.utils.config import Config, SliceType


class SceneEntityCfg(Config):
    """The configuration for the scene entity.

    A scene entity can be a robot, a tool, or any other object in the scene.
    This configuration is used to specify the scene entity and the required
    joints, bodies, and fixed tendons from the scene entity.


    Note:
        User should resolve the joint names, body names, and fixed tendon names
        to indices on post initialization.

        User should implement the :meth:`resolve` method to resolve the names
        to indices.

    """

    name: str
    """The name of the scene entity.

    This is the name defined in the scene configuration file.
    """

    joint_names: str | list[str] | None = None
    """The names of the joints from the scene entity. Defaults to None.

    The names can be either joint names or a regular expression matching
    the joint names.

    Users should convert the joint names to joint indices under
    :attr:`joint_ids` on post initialization.
    """

    joint_ids: list[int] | SliceType = slice(None)
    """The indices of the joints from the asset required by the term.
    Defaults to slice(None), which means all the joints in the
    asset (if present).

    If :attr:`joint_names` is specified, this field should be filled in
    automatically on post initialization.
    """

    fixed_tendon_names: str | list[str] | None = None
    """The names of the fixed tendons from the scene entity. Defaults to
    None. The names can be either joint names or a regular expression matching
    the joint names.

    Users should convert this field to indices under :attr:`fixed_tendon_ids`
    on post initialization.
    """

    fixed_tendon_ids: list[int] | SliceType = slice(None)
    """The indices of the fixed tendons from the asset required by the term.
    Defaults to slice(None), which means all the fixed tendons in the
    asset (if present).

    If :attr:`fixed_tendon_names` is specified, this field should be filled in
    automatically on post initialization.
    """

    body_names: str | list[str] | None = None
    """The names of the bodies from the asset required by the term. Defaults
    to None. The names can be either body names or a regular expression
    matching the body names.

    Users should convert this field to indices under :attr:`body_ids`
    on post initialization.
    """

    body_ids: list[int] | SliceType = slice(None)
    """The indices of the bodies from the asset required by the term.
    Defaults to slice(None), which means all the bodies in the asset.

    If :attr:`body_names` is specified, this field should be filled in
    automatically on post initialization.
    """

    preserve_order: bool = False
    """Whether to preserve indices ordering to match with that in the specified
    joint or body names. Defaults to False.

    If False, the ordering of the indices are sorted in ascending order
    (i.e. the ordering in the entity's joints or bodies). Otherwise, the
    indices are preserved in the order of the specified joint and body names.


    For more details, see:
        :meth:`robo_orchard.utils.string.resolve_matching_names`

    Note:
        This attribute is only used when :attr:`joint_names` or
        :attr:`body_names` are specified.

    """

    @abstractmethod
    def resolve(self, scene: Any):
        """Resolve all names to indices.

        This function should be implemented by the user to resolve the joint
        names, body names, and fixed tendon names to indices.

        Args:
            scene: The scene object to resolve the names to indices.

        """
        raise NotImplementedError
