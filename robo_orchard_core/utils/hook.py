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

from __future__ import annotations
import weakref
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Generator, Generic, TypeVar

from ordered_set import OrderedSet
from typing_extensions import Self

T = TypeVar("T")
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


class HookContext(Generic[T], metaclass=ABCMeta):
    """Context manager for executing hooks with a specific set of arguments.

    This class is used to manage the lifecycle of hooks, ensuring that
    the appropriate hook methods are called at the right times. It also
    provides a way to pass arguments to the hooks in a structured manner.
    """

    @abstractmethod
    def before(self, arg: T):
        """Prepare the context for executing hooks.

        This method should set up the context for executing hooks and
        return the arguments referenced by the hooks.

        The returned arguments will be used in the `after` method to clean up
        the context.

        Args:
            arg (T): The arguments to be passed to the hooks.
        """
        raise NotImplementedError("Subclasses must implement before method.")

    @abstractmethod
    def after(self, arg: T):
        """Clean up the context after executing hooks.

        This method should clean up any resources or state used during
        the execution of hooks.

        Args:
            arg (T): The arguments to be passed to the hooks.

        """
        raise NotImplementedError("Subclasses must implement after method.")

    @contextmanager
    def begin(self, arg: T) -> Generator[T, Any, None]:
        """Context manager for executing hooks with given arguments.

        This method sets up the context for executing hooks and yields
        the arguments to be passed to the hooks. After the hooks are executed,
        the context is cleaned up.
        """
        try:
            self.before(arg)
            yield arg
        finally:
            self.after(arg)

    @staticmethod
    def from_callable(
        before: Callable[[T], None] | None = None,
        after: Callable[[T], None] | None = None,
    ) -> HookContext[T]:
        """Create a hook context from a callable.

        Args:
            before (Callable[[T], None]): The callable to be used as the
                `before` method.
            after (Callable[[T], None]): The callable to be used as the
                `after` method.

        Returns:
            HookContext[T]: The hook context created from the callable.
        """
        return HookContextFromCallable(before, after)


class HookContextFromCallable(HookContext[T]):
    """A hook context that is created from a callable.

    This class is used to create a hook context from a callable that
    takes a single argument and returns a value. The callable is called
    in the `before` method and the returned value is passed to the
    `after` method.

    Args:
        func (Callable[[T], T]): The callable to be used as the hook context.
    """

    def __init__(
        self,
        before: Callable[[T], None] | None = None,
        after: Callable[[T], None] | None = None,
    ):
        """Initialize the hook context with a callable."""
        self._before = before
        self._after = after

    def before(self, arg: T):
        if self._before is not None:
            self._before(arg)

    def after(self, arg: T):
        if self._after is not None:
            self._after(arg)


class HookContextChannel(Generic[T]):
    """A channel for managing hook context.

    The hook contexts are registered and executed in a specific order
    when the `begin` method is called. This class provides a way to
    register and unregister hook context handlers.

    """

    def __init__(self, name: str):
        self.name = name
        self._context_handlers: OrderedSet[HookContext[T]] = OrderedSet([])

    def __len__(self) -> int:
        """Get the number of registered hook context handlers."""
        return len(self._context_handlers)

    def register(self, hook: HookContext[T]):
        """Register a hook context handler.

        Args:
            hook (HookContext[T]): The hook context handler to register.

        Returns:
            RemoveableHandle: A handle to remove the registered hook.
        """
        self._context_handlers.add(hook)
        return RemoveableHandle(lambda: self._context_handlers.discard(hook))

    def register_hook_channel(
        self,
        channel: HookContextChannel[T],
    ) -> RemoveableHandle[Callable[[], None]]:
        """Register a hook context channel.

        Args:
            channel (HookContextChannel[T]): The hook context channel
                to register.

        Returns:
            RemoveableHandle: A handle to remove the registered hook.
        """
        if channel.name != self.name:
            raise ValueError(
                f"Cannot register hook channel {channel.name} to {self.name}"
            )
        to_remove = []
        for hook in channel._context_handlers:
            self._context_handlers.add(hook)
            to_remove.append(hook)

        def remove():
            for hook in to_remove:
                self._context_handlers.discard(hook)

        return RemoveableHandle(remove)

    def __iadd__(self, other: HookContextChannel[T] | HookContext[T]) -> Self:
        """Add another set of pipeline hooks to the current instance.

        Args:
            other (HookContextChannel[T] | HookContext[T]):
                The other set of pipeline hooks to add.
                It can be either a HookContextChannel or a HookContext.

        Returns:
            Self: The current instance with the added hooks.
        """
        if isinstance(other, HookContextChannel):
            self.register_hook_channel(other)
        elif isinstance(other, HookContext):
            self.register(other)
        else:
            raise TypeError(
                f"Cannot add {type(other)} to {type(self)}. "
                "Expected HookContextChannel or HookContext."
            )
        return self

    def unregister_all(self):
        """Unregister all hook context handlers."""
        self._context_handlers.clear()

    @contextmanager
    def begin(self, arg: T):
        """Context manager for executing hooks with given arguments.

        This method sets up the context for executing hooks and yields
        the arguments to be passed to the hooks. After the hooks are executed,
        the context is cleaned up.

        """
        try:
            for hook in self._context_handlers:
                hook.before(arg)
            yield arg
        finally:
            for hook in self._context_handlers:
                hook.after(arg)
