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

from __future__ import annotations
import typing
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Generic

from typing_extensions import TypeVar

from robo_orchard_core.datatypes.dataclass import DataClass
from robo_orchard_core.envs.env_base import EnvBase
from robo_orchard_core.envs.managers.manager_term_base import (
    ManagerTermBase,
    ManagerTermBaseCfg,
)
from robo_orchard_core.envs.managers.scene_entity_cfg import SceneEntityCfg

EventTermType_co = TypeVar(
    "EventTermType_co", bound="EventTermBase", covariant=True
)

EventCfgType_co = TypeVar(
    "EventCfgType_co",
    bound="EventTermBaseCfg",
    covariant=True,
    default="EventTermBaseCfg",
)
EnvType_co = TypeVar(
    "EnvType_co", bound=EnvBase, covariant=True, default=EnvBase
)


class EventMsg(DataClass):
    """The base message for the event term to trigger the event."""

    pass


#:
EventMsgType = TypeVar("EventMsgType", bound=EventMsg)


class EventTermBase(
    ManagerTermBase[EnvType_co, EventCfgType_co],
    Generic[EventMsgType, EnvType_co, EventCfgType_co],
    metaclass=ABCMeta,
):
    """The base class for all event terms.

    A event term is a term that applies operations to the scene entities in the
    environment or the environment itself. Different from action terms that
    sent from agents to interact with the environment, event terms are usually
    triggered by the environment itself, or operations that are not directly
    controlled by agents.

    For example, a event term can be a term that changes the color of the scene
    entities when the environment reaches a certain state, or a term that
    applies external forces to the scene entities.

    Note that the event term can access the environment and the scene entities
    in the environment, which means that any callble method of the environment
    and the scene entities can be called in the event term.

    Template Args:
        EventMsgType: The type of the event message.
        EnvType_co: The type of the environment.
        EventCfgType: The configuration of the event term.

    Args:
        cfg: The configuration of the event term.
        env: The environment.

    """

    event_msg_type: type[EventMsgType]

    @classmethod
    def __init_subclass__(cls) -> None:
        """Set the event_msg_type and target_type class attributes.

        Reference:
            https://peps.python.org/pep-0487/

        """
        type_args = typing.get_args(cls.__orig_bases__[0])  # type: ignore
        if type_args == () or len(type_args) < 1:
            raise ValueError(
                f"EventTermBase subclass must have at least one type argument."  # noqa
            )

        cls.event_msg_type = type_args[0]

    def __init__(self, cfg: EventCfgType_co, env: EnvType_co):
        super().__init__(cfg, env)

    @abstractmethod
    def __call__(self, event_msg: EventMsgType):
        """The implementation of the event term.

        All subclasses should implement this method to apply the operations to
        the scene entities or the environment.

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the action term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        """
        raise NotImplementedError


EntityCfgType_co = TypeVar(
    "EntityCfgType_co",
    bound=SceneEntityCfg,
    covariant=True,
    default=SceneEntityCfg,
)


class EventTermBaseCfg(
    ManagerTermBaseCfg[EventTermType_co],
    Generic[EventTermType_co, EntityCfgType_co],
):
    """The base configuration for all event terms.

    Template Args:
        EventTermType_co: The type of the event term.
        EntityCfgType_co: The type of the scene entity configuration.

    """

    trigger_topic: str
    """The topic to trigger the event term."""

    asset_cfgs: list[EntityCfgType_co] | None = None
    """The scene entitys that the event term is associated with.

    If None, the event term is associated with all scene entitys.
    """
