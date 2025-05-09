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

"""Timer utilities for timing function execution."""

import logging
import types
from inspect import isfunction
from time import perf_counter
from typing import Any, Optional

from robo_orchard_core.utils.logging import LoggerManager

logger = LoggerManager().get_child(__name__)


class Timer(object):
    """Timer class for timing function execution.

    Modify from https://github.com/LucienShui/timer .


    Args:
        name_or_func (Any): name of function, or function itself,
            user should not care about this parameter. Defaults to None.
        unit (str): time's unit, should be one of 's', 'ms' or 'auto'.
            Defaults to 'auto'.
        logger (logging.Logger): logger object. Defaults to logger.

    Example:
    >>> with Timer():
    ...     print("hello")
    cost 0.001 s

    >>> @Timer()
    ... def foo():
    ...     print("hello")
    >>> foo()
    cost 0.001 s

    >>> @Timer("foo")
    ... def bar():
    ...     print("hello")
    >>> bar()
    cost 0.001 s

    """

    _level = logging.DEBUG

    def __init__(
        self,
        name_or_func: Any = None,
        unit: str = "auto",
        logger: logging.Logger = logger,
    ):
        if unit not in ["s", "ms", "auto"]:
            raise AssertionError(
                f"field unit should be one of 's', 'ms', 'auto', got {unit}"
            )

        if isfunction(name_or_func):
            self._func = name_or_func
            self._name: Optional[str] = None
        else:
            self._func = None
            self._name: Optional[str] = name_or_func

        self._unit: str = unit

        self._logger: logging.Logger = logger
        self._begin: float = ...  # type: ignore
        self._end: float = ...  # type: ignore

    def _log(self, message: str, name: Optional[str] = None) -> None:
        def _log_impl(logger_: logging.Logger, level: int, message: str):
            if level == logging.DEBUG:
                logger_.debug(message)
            elif level == logging.INFO:
                logger_.info(message)
            elif level == logging.WARNING:
                logger_.warning(message)
            elif level == logging.ERROR:
                logger_.error(message)
            elif level == logging.CRITICAL:
                logger_.critical(message)
            else:
                raise AssertionError("wrong level")

        if self._level is not None:
            if name is not None:
                _log_impl(self._logger.getChild(name), self._level, message)

    @property
    def elapse(self) -> float:
        """Return the time cost in seconds."""
        if self._end is ...:
            end = perf_counter()
        else:
            end = self._end
        return end - self._begin

    def _start(self, name: str) -> None:
        self._log("start", name=name)
        self._begin = perf_counter()

    def _stop(self, name: str) -> None:
        self._end = perf_counter()
        if self._unit == "ms" or (self._unit == "auto" and self.elapse < 1000):
            self._log(f"cost {self.elapse * 1000:.3f} ms", name=name)
        else:
            self._log(f"cost {self.elapse:.3f} s", name=name)

    def __enter__(self):
        self._start(self._name or "timer")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop(self._name or "timer")

    def __get__(self, instance, owner):
        """Return object itself when decorate function of object.

        Args:
            instance (Any): ?
            owner (Any): ?

        """
        if instance is None:
            return self
        return types.MethodType(self, instance)

    def __call__(self, *args, **kwargs):
        if self._func is None:
            func = args[0]

            def wrapper(*_args, **_kwargs):
                __name: str = self._name or func.__name__
                self._start(__name)
                _result = func(*_args, **_kwargs)
                self._stop(__name)
                return _result

            return wrapper
        else:
            name: str = self._name or self._func.__name__
            self._start(name)
            result = self._func(*args, **kwargs)
            self._stop(name)
            return result

    @classmethod
    def set_level(cls, new_level: int):
        cls._level = new_level


class FPSCounter:
    """A simple class to count the frames per second."""

    def __init__(self, window_size_sec: float = 1):
        self._window_size_sec = window_size_sec
        self._frame_counter = 0
        self._last_time = 0
        self._fps = 0

        self._half_last_time = 0
        self._half_frame_counter = 0

    def reset(self):
        """Reset the frame counter."""
        self._frame_counter = 0
        self._last_time = 0
        self._fps = 0

        self._half_last_time = 0
        self._half_frame_counter = 0

    @property
    def fps(self) -> float:
        return self._fps

    def update(self):
        """Update the frame counter."""
        current_time = perf_counter()

        # reset for every second
        if current_time - self._last_time >= self._window_size_sec:
            self._last_time = self._half_last_time
            self._frame_counter = (
                self._frame_counter - self._half_frame_counter
            )

        if current_time - self._half_last_time >= self._window_size_sec / 2:
            self._half_last_time = current_time
            self._half_frame_counter = self._frame_counter

        self._frame_counter += 1

        if self._last_time == 0:
            self._last_time = current_time
            return

        self._fps = (self._frame_counter - 1) / (
            current_time - self._last_time
        )
