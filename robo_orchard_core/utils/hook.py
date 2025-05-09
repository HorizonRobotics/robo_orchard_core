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

"""The hook utilities for registering and calling hooks."""

import weakref
from typing import Any, Callable, Generic, TypeVar

from ordered_set import OrderedSet

CallableType = TypeVar("CallableType", bound=Callable)


class RemoveableHandle(Generic[CallableType]):
    """A handle that can be removed.

    Args:
        hooks (Callable): The hook to call.
            Be careful that the hook may cause circular reference.
    """

    def __init__(self, hook: CallableType):
        self.hook: CallableType | None = hook

    def remove(self):
        """Remove the handle."""
        self.hook = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.hook is not None:
            return self.hook(*args, **kwds)
        else:
            raise RuntimeError("The handle has been removed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


class HookHandler(Generic[CallableType]):
    """The hook handler.

    Args:
        name (str): The name of the hook.

    """

    def __init__(self, name: str):
        self.name = name
        self._handlers: OrderedSet[CallableType] = OrderedSet()  # type: ignore

    def __call__(self, *args, **kwargs):
        # use copy to prevent  RuntimeError: Set changed size during iteration
        for handler in self._handlers.copy():
            handler(*args, **kwargs)

    def __len__(self) -> int:
        return len(self._handlers)

    def register(
        self, handler: CallableType
    ) -> RemoveableHandle[Callable[[], None]]:
        """Register a handler.

        Args:
            handler (Callable): The handler to register.

        Returns:
            RemoveableHandle: The removeable handle.
        """
        self._handlers.add(handler)
        return RemoveableHandle(lambda: self._handlers.discard(handler))

    def register_once(
        self, handler: CallableType
    ) -> RemoveableHandle[Callable[[], None]]:
        """Register a handler that will be called only once.

        Args:
            handler (Callable): The handler to register.

        Returns:
            RemoveableHandle: The removeable handle.
        """

        def _handler(*args, **kwargs):
            handler(*args, **kwargs)
            weakref.proxy(self._handlers).remove(_handler)

        self._handlers.add(_handler)  # type: ignore
        return RemoveableHandle(
            lambda: self._handlers.discard(_handler)  # type: ignore
        )

    def unregister_all(self):
        """Unregister all handlers."""
        self._handlers.clear()

    def __repr__(self):
        return f"<HookHandler {self.name}>({len(self._handlers)} handlers: {self._handlers})"  # noqa
