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

"""The registry module for registering and calling objects."""

from typing import Any, Callable

__all_ = ["Registry"]


class Registry(object):
    """The registry that provides name -> object mapping.

    Registry can be widely used in Factory Design Pattern. It is a global
    dictionary that stores the mapping from object names to objects. It allows
    to register objects by name, and then create objects by name.

    But abuse of registry is not recommended. It makes the code hard to
    understand. It is recommended to use registry only when necessary.

    .. code-block:: python

        BACKBONE_REGISTRY = Registry("BACKBONE")

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register
        class MyBackbone: ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)

    To register an object with alias :

    .. code-block:: python

        @BACKBONE_REGISTRY.register
        @BACKBONE_REGISTRY.alias("custom")
        class MyBackbone: ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone, "custom")

    To get an object:

    .. code-block:: python

        backbone = BACKBONE_REGISTRY.get("MyBackbone")


    Args:
        name (str): The name of the registry.
        case_sensitive (bool, optional): Whether the name is case sensitive.
            Defaults to True.

    """

    def __init__(self, name, case_sensitive=True):
        self._name = name
        self._case_sensitive = case_sensitive
        self._name_obj_map = {}

    def pop(self, name: str, default: Any = None) -> Any:
        """Remove the object with name and return it.

        Args:
            name (str): The name of the object.
            default: The default value if the object is not found.
                Default is None.

        """
        return self._name_obj_map.pop(name, default)

    def __contains__(self, name):
        if isinstance(name, str) and not self._case_sensitive:
            name = name.lower()
        return name in self._name_obj_map

    def _do_register(self, name, obj):
        if isinstance(name, str) and not self._case_sensitive:
            name = name.lower()
        if name in self._name_obj_map:
            raise ValueError(
                f"An object named '{name}' was already registered "
                f"in '{self._name}' registry!"
            )
        self._name_obj_map[name] = obj

    def keys(self):
        """Get the names of the registered objects."""
        return self._name_obj_map.keys()

    def values(self):
        """Get the registered objects."""
        return self._name_obj_map.values()

    def register(self, obj: Any = None, *, name: str | None = None) -> Any:
        """Regist an object.

        Register the given object under the the name `obj.__name__`
        or given name.

        Args:
            obj (Any): The object to register. If None, it will be used
                as a decorator.
            name (str, optional): The name of the object. Defaults to None.

        """
        if obj is None and name is None:
            raise ValueError("Should provide at least one of obj and name")
        if obj is not None and name is not None:
            self._do_register(name, obj)
        elif obj is not None and name is None:  # used as decorator
            name = obj.__name__
            self._do_register(name, obj)
            return obj
        else:
            return self.alias(name)

    def alias(self, name) -> Callable:
        """Get registrator function that allow aliases.

        Args:
            name (str): The name of the object.

        Returns:
            Callable: The registrator function.
        """

        def reg(obj):
            self._do_register(name, obj)
            return obj

        return reg

    def get(self, name: str, raise_not_exist: bool = True):
        """Get the object with name.

        Args:
            name (str): The name of the object.
            raise_not_exist (bool, optional): Whether to raise an error if the
                object is not found. Defaults to True.

        Returns:
            Any: The object.
        """

        origin_name = name
        if isinstance(name, str) and not self._case_sensitive:
            name = name.lower()
        ret = self._name_obj_map.get(name)
        if ret is None:
            if raise_not_exist:
                raise KeyError(
                    "No object named '{}' found in '{}' registry!".format(
                        origin_name, self._name
                    )
                )
        return ret

    def __getitem__(self, name):
        return self.get(name)
