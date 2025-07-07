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

from typing import Any, Callable, Dict, Generic, List, Mapping, Sequence, Type

from typing_extensions import TypeVar

from robo_orchard_core.envs.managers.events.event_term import (
    EventMsg,
    EventMsgType,
    EventTermBase,
    EventTermBaseCfg,
)
from robo_orchard_core.envs.managers.manager_base import (
    EnvType_co,
    ManagerBase,
    ManagerBaseCfg,
)
from robo_orchard_core.utils.config import ClassType_co
from robo_orchard_core.utils.hook import HookHandler, RemoveableHandle
from robo_orchard_core.utils.logging import LoggerManager

logger = LoggerManager().get_child(__name__)

EventTermConfigType_co = TypeVar(
    "EventTermConfigType_co",
    bound=EventTermBaseCfg,
    covariant=True,
    default=EventTermBaseCfg,
)

EventManagerConfigType_co = TypeVar(
    "EventManagerConfigType_co", bound="EventManagerCfg", covariant=True
)


class UnregisteredEventChannelError(Exception):
    """Raised when an unregistered event channel is accessed."""

    def __init__(self, topic: str):
        self.topic = topic
        super().__init__(
            f"Event channel with topic {topic} is not registered."
        )


class EventChannel(
    HookHandler[EventTermBase[EventMsgType, Any, Any]], Generic[EventMsgType]
):
    """The event channel for publishing and subscribing to events.

    The event channel is a communication channel for publishing and subscribing
    to events. The event channel is identified by a topic. The subscribers can
    subscribe to the event channel by providing a callback function. Once the
    event is published, the subscribers will be called in the order they
    subscribed.

    Template Args:
        EventMsgType: The type of the event message.

    Args:
        topic (str): The topic of the event channel.

    """

    def __init__(self, topic: str, event_msg_type: type[EventMsgType]):
        super().__init__(name=topic)
        self._event_msg_type = event_msg_type

    def subscribe(
        self, term: EventTermBase[EventMsgType, Any, Any]
    ) -> RemoveableHandle[Callable[[], None]]:
        """Subscribe to the event channel.

        Args:
            term (EventTermBase): The term to subscribe.

        Returns:
            The callback function to unsubscribe.

        Raises:
            ValueError: If the term's event message type does not match the
                event channel's event message type.

        """
        if term.event_msg_type != self._event_msg_type:
            raise ValueError(
                f"The term's event message type {term.event_msg_type} "
                f"does not match the event channel's event message type "
                f"{self._event_msg_type}."
            )

        return self.register(term)

    def notify(self, msg: EventMsgType) -> None:
        """Notify the subscribers of the event channel.

        The message will be sent to all the subscribers. The subscribers
        will be called in the order they subscribed.

        Args:
            msg (EventMsgType): The message to publish.

        """
        return self.__call__(msg)

    def unsubscribe_all(self) -> None:
        """Unsubscribe all the subscribers."""
        self.unregister_all()


class EventChannelHub:
    """The event channel hub for managing event channels."""

    def __init__(self):
        self._channels: dict[str, EventChannel] = {}

    @property
    def topics(self) -> set[str]:
        """Return the topics of the registered event channels."""
        return set(self._channels.keys())

    def register_channel(
        self, topic: str, msg_type: Type[EventMsgType]
    ) -> EventChannel[EventMsgType]:
        """Register an event channel.

        Args:
            topic (str): The topic of the event channel.
            msg_type (Type[EventMsgType]): The type of the event message.

        Returns:
            EventChannel: The event channel.

        Raises:
            ValueError: If msg_type is not a type.

        """

        # check that msg_type is a type
        if not isinstance(msg_type, type):
            raise ValueError(
                f"msg_type must be a type, got {msg_type} instead."
            )
        if topic not in self._channels:
            logger.info(
                f"Registering event channel with topic {topic} "
                f"with message type {msg_type}"
            )
            self._channels[topic] = EventChannel[EventMsgType](
                topic, event_msg_type=msg_type
            )
        return self._channels[topic]

    def unregister_channel(self, topic: str) -> EventChannel | None:
        """Unregister an event channel."""
        if topic in self._channels:
            return self._channels.pop(topic)
        return None

    def subscribe(
        self,
        topic: str,
        term: EventTermBase[EventMsgType, Any, Any],
        auto_register: bool = False,
    ) -> Callable[[], None]:
        """Subscribe to an event channel.

        Args:
            topic (str): The topic of the event channel.
            term (EventTermBase): The term to subscribe.
            auto_register (bool): Whether to automatically register the event
                channel if it does not exist. Defaults to False.

        Returns:
            Callable[[], None]: The callback function to unsubscribe.

        Raises:
            UnregisteredEventChannelError: If the event channel is not
                registered and auto_register is False.

        """

        channel: EventChannel[EventMsgType] | None = self._channels.get(topic)
        if channel is None:
            if auto_register:
                channel = self.register_channel(
                    topic, msg_type=term.event_msg_type
                )
            else:
                raise UnregisteredEventChannelError(topic)

        return channel.subscribe(term)

    def notify(self, topic: str, msg: EventMsg) -> None:
        """Notify the subscribers of the event channel.

        Args:
            topic (str): The topic of the event channel.
            msg (EventMsgType): The message to publish.

        Raises:
            UnregisteredEventChannelError: If the event channel is not
                registered.

        """

        channel: EventChannel[EventMsg] | None = self._channels.get(topic)
        if channel is None:
            raise UnregisteredEventChannelError(topic)

        channel.notify(msg)

    def unsubscribe(self, topic: str | None) -> None:
        """Unsubscribe all the subscribers of the event channel.

        Args:
            topic (str | None): The topic of the event channel. If None, all
                event channels will be unsubscribed.
        """
        if topic is not None:
            channel: EventChannel | None = self._channels.get(topic)
            if channel is not None:
                channel.unsubscribe_all()
        else:
            for channel in self._channels.values():
                channel.unsubscribe_all()


class EventManager(ManagerBase[EnvType_co, EventManagerConfigType_co]):
    """The event manager for managing event terms."""

    def __init__(
        self,
        cfg: EventManagerConfigType_co,
        env: EnvType_co,
    ):
        super().__init__(cfg, env)
        self._term_cfgs: Mapping[str, EventTermBaseCfg] = self.cfg.terms
        self._term_names: List[str] = list(self._term_cfgs.keys())
        self._terms: Dict[str, EventTermBase] = self.cfg.create_terms(env)
        self._channel_hub = EventChannelHub()
        self._subscribe_terms()

    def _subscribe_terms(self) -> None:
        for term_name, term in self._terms.items():
            term_cfg = self._term_cfgs[term_name]
            self._channel_hub.subscribe(
                term_cfg.trigger_topic, term, auto_register=True
            )

    def register(
        self, topic: str, msg_type: Type[EventMsgType]
    ) -> EventChannel[EventMsgType]:
        """Register an event channel.

        Args:
            topic (str): The topic of the event channel.
            msg_type (Type[EventMsgType]): The type of the event message.

        Returns:
            EventChannel: The event channel.

        Raises:
            ValueError: If msg_type is not a type.

        """
        return self._channel_hub.register_channel(topic, msg_type)

    @property
    def active_terms(self) -> List[str]:
        """Returns the active terms."""
        return self._term_names

    @property
    def event_topics(self) -> set[str]:
        """Returns the event topics."""
        return self._channel_hub.topics

    def notify(self, topic: str, msg: Any) -> None:
        """Notify the subscribers of the event channel.

        Args:
            topic (str): The topic of the event channel.
            msg (EventMsgType): The message to publish.

        """
        self._channel_hub.notify(topic, msg)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        for term in self._terms.values():
            term.reset(env_ids=env_ids)


class EventManagerCfg(
    ManagerBaseCfg[EventManager],
    Generic[EventTermConfigType_co],
):
    """The configuration for the event manager.

    Template Args:
        EventTermConfigType_co: The type of the event term configuration.
    """

    class_type: ClassType_co[EventManager] = EventManager

    terms: Mapping[str, EventTermConfigType_co] = {}

    def create_terms(self, env: Any) -> Dict[str, EventTermBase]:
        return {
            term_name: term_cfg(env=env)
            for term_name, term_cfg in self.terms.items()
        }
