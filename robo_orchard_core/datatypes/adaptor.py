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


"""Data type adaptors for converting objects of one type to another."""

from __future__ import annotations
import typing
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, Sequence, TypeVar

from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType_co,
)
from robo_orchard_core.utils.registry import Registry

__all__ = [
    "TypeAdaptorImpl",
    "TypeAdaptorImplConfig",
    "TypeAdaptorImplConfigType_co",
    "TypeAdaptorFactory",
    "TypeAdaptorFactoryConfig",
    "UnRegisteredAdaptorError",
]

S = TypeVar("S")
T = TypeVar("T")


class UnRegisteredAdaptorError(Exception):
    pass


def _type_to_str(t: type) -> str:
    return f"{t.__module__}:{t.__name__}"


class TypeAdaptorImpl(
    ClassInitFromConfigMixin, Generic[S, T], metaclass=ABCMeta
):
    """A generic class that defines the interface for type adaptors.

    Type adaptors are used to convert objects of one type to another.
    For example, a type adaptor can convert a custom object to ROS message.

    User should implement the __call__ method to define the conversion logic.

    Example:
        >>> class MyTypeAdaptor(TypeAdaptorImpl[Source, Target]):
        ...     def __call__(self, s: Source) -> Target:
        ...         # Conversion logic
        ...         pass

    """

    source_type: type[S]
    """The source type of the adaptor to take as input."""
    target_type: type[T]
    """The target type of the adaptor as output."""

    @classmethod
    def __init_subclass__(cls) -> None:
        """Set the source_type and target_type class attributes.

        Reference:
            https://peps.python.org/pep-0487/

        """
        type_args = typing.get_args(cls.__orig_bases__[0])  # type: ignore
        if type_args == () or len(type_args) != 2:
            raise ValueError(
                "TypeAdaptorImpl should have two type arguments: "
                "source_type and target_type. "
                f"Gotten: {type_args}. "
                """Example:
class MyTypeAdaptor(TypeAdaptorImpl[Source, Target]):
    pass
                """
            )

        cls.source_type = type_args[0]
        cls.target_type = type_args[1]

    @abstractmethod
    def __call__(self, s: S) -> T:
        pass


class TypeAdaptorImplConfig(ClassConfig[TypeAdaptorImpl], Generic[S, T]):
    """Config class for TypeAdaptorImpl.

    Template Args:
        S: The source type of the adaptor.
        T: The target type of the adaptor.

    """

    class_type: ClassType_co[TypeAdaptorImpl]


TypeAdaptorImplConfigType_co = TypeVar(
    "TypeAdaptorImplConfigType_co", bound=TypeAdaptorImplConfig, covariant=True
)


class TypeAdaptorFactory(ClassInitFromConfigMixin):
    """TypeAdaptorFactory is a collection of TypeAdaptorImpl instances.

    It provides type adaptors for different types of input objects and
    converts them to the target type. If the type adaptor for the input object
    is not found, it will raise an error when skip_unregistered is False,
    otherwise it will return the input object.

    Example:
        .. code-block:: python

            factory_cfg = TypeAdaptorFactoryConfig(
                class_type=TypeAdaptorFactory,
                adaptors=[
                    TypeAdaptorImplConfig(class_type=Int2StrAdaptor),
                    TypeAdaptorImplConfig(class_type=List2StrAdaptor),
                ],
            )

            factory = TypeAdaptorFactory(factory_cfg)
            assert factory(1) == "1"
            assert factory([1, 2, 3]) == "[1, 2, 3]"

    Args:
        cfg (TypeAdaptorFactoryConfig): The config object for the factory.

    """

    def __init__(self, cfg: TypeAdaptorFactoryConfig | None = None):
        if cfg is None:
            cfg = TypeAdaptorFactoryConfig()
        self._cfg = cfg
        self._skip_unregistered = cfg.skip_unregistered
        self._adaptors = Registry("adaptors")
        for adaptor in cfg.adaptors:
            self.register(adaptor())

    def get_source_type_adaptor(
        self, source_type: type[S]
    ) -> TypeAdaptorImpl[S, Any] | None:
        """Get the type adaptor for the given source type.

        Args:
            source_type (type): The source type to get the adaptor for.

        Returns:
            TypeAdaptorImpl | None: The type adaptor for the given source type.

        """

        return self._adaptors.get(
            _type_to_str(source_type), raise_not_exist=False
        )

    def unregister(self, source_type: type[S]):
        """Unregister the type adaptor for the given source type.

        Args:
            source_type (type): The source type to unregister the adaptor for.

        """
        self._adaptors.pop(_type_to_str(source_type))

    def register(self, adaptor: TypeAdaptorImpl):
        """Register a type adaptor.

        Args:
            adaptor (TypeAdaptorImpl): The type adaptor to register.

        """

        source_type = adaptor.source_type
        self._adaptors.register(obj=adaptor, name=_type_to_str(source_type))

    def __call__(self, s):
        """Convert the input object to the target type.

        If the type adaptor for the input object is not found, it will raise
        an error when skip_unregistered is False, otherwise it will return the
        input object.

        Args:
            s (Any): The input object to convert.

        Returns:
            Any: The converted object.

        Raises:
            UnRegisteredAdaptorError: If the type adaptor is not found and
                skip_unregistered is False.
        """
        source_type_adaptor = self.get_source_type_adaptor(type(s))
        if source_type_adaptor is None:
            if self._skip_unregistered:
                return s
            raise UnRegisteredAdaptorError(
                f"Cannot find adaptor for {type(s)}"
            )
        return source_type_adaptor(s)

    @property
    def source_types(self) -> set[type]:
        """Get the set of source types that the factory can convert."""
        ret = []
        for adaptor in self._adaptors.values():
            ret.append(adaptor.source_type)
        return set(ret)


class TypeAdaptorFactoryConfig(ClassConfig[TypeAdaptorFactory]):
    """Config class for TypeAdaptorFactory."""

    class_type: ClassType_co[TypeAdaptorFactory] = TypeAdaptorFactory
    adaptors: Sequence[TypeAdaptorImplConfigType_co] = []  # type: ignore
    skip_unregistered: bool = False
