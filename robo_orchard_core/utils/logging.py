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

"""The logging module."""

import logging
import sys

from typing_extensions import Dict, Optional, Self

DEFAULT_LOG_FORMAT = (
    "%rank %(asctime)-15s %(levelname)s "
    "| %(process)d | %(threadName)s | "
    "%(name)s:L%(lineno)d %(message)s"
)


def wrap_log_fmt_with_rank(format: str) -> str:
    """Wrap the log format with the rank of the process."""
    from robo_orchard_core.utils.distributed import get_dist_info

    if "%rank" in format:
        dist_info = get_dist_info()
        format = format.replace(
            "%rank", "Rank[{}/{}]".format(dist_info.rank, dist_info.world_size)
        )
    return format


class _Singleton:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance


class LoggerManager(_Singleton):
    """A logger manager that manages the logger.

    This class is a singleton class that manages the logger. It provides
    methods to get the logger and its child logger. You can use the logger
    manager to handle all loggers in the project.

    Example:

    .. code-block:: python

        # To get a child logger
        from robo_orchard_core.utils.logging import LoggerManager

        logger = LoggerManager().get_child(__name__)


    Args:
        format (str, optional): The format of the log. Defaults to
            DEFAULT_LOG_FORMAT.
        level (int, optional): The level of the log. Defaults to logging.INFO.
        handlers (Optional[list[logging.Handler]], optional): The handlers of
            the log. Defaults to None.

    """

    def __init__(
        self,
        format: str = DEFAULT_LOG_FORMAT,
        level: int = logging.INFO,
        handlers: Optional[list[logging.Handler]] = None,
    ):
        self._logger = logging.getLogger("LoggerManager")
        self._logger.propagate = False
        self._logger.setLevel(level)
        self._format = wrap_log_fmt_with_rank(format)
        self._level = level
        if handlers is None:
            handlers = [
                logging.StreamHandler(sys.stdout),
            ]
        self.set_handlers(handlers)
        self._child_loggers: Dict[str, logging.Logger] = {}

    def get_logger(self) -> logging.Logger:
        """Get the global logger."""
        return self._logger

    def get_child(self, name: str) -> logging.Logger:
        """Get the child logger.

        Args:
            name (str): The name of the child logger.

        Returns:
            logging.Logger: The child logger.

        """
        ret = self._logger.getChild(name)
        self._child_loggers[name] = ret
        return ret

    def set_level(self, level: int, recursive: bool = False) -> Self:
        """Set the level of the logger.

        Args:
            level (int): The level of the logger.
            recursive (bool, optional): Whether to set the level recursively.
                Defaults to False.

        Returns:
            Self: The logger manager.

        """

        self._level = level

        loggers_to_set = [self._logger]
        if recursive:
            loggers_to_set.extend(self._child_loggers.values())

        for logger in loggers_to_set:
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)

        return self

    def set_format(self, format: str, recursive: bool = False) -> Self:
        """Set the format of the logger.

        Args:
            format (str): The format of the logger.
            recursive (bool, optional): Whether to set the format recursively.
                Defaults to False.

        Returns:
            Self: The logger manager.

        """
        format = wrap_log_fmt_with_rank(format)
        self._format = format

        loggers_to_set = [self._logger]
        if recursive:
            loggers_to_set.extend(self._child_loggers.values())

        for logger in loggers_to_set:
            for handler in logger.handlers:
                handler.setFormatter(logging.Formatter(format))

        return self

    def set_handlers(
        self, handlers: list[logging.Handler], recursive: bool = False
    ) -> Self:
        """Set the handlers of the logger.

        Args:
            handlers (list[logging.Handler]): The handlers to set.
            recursive (bool, optional): Whether to set the handlers
                recursively. Defaults to False.

        Returns:
            Self: The logger manager.
        """

        loggers_to_set = [self._logger]
        if recursive:
            loggers_to_set.extend(self._child_loggers.values())

        for logger in loggers_to_set:
            logger.handlers.clear()
            for handler in handlers:
                handler.setFormatter(logging.Formatter(self._format))
                handler.setLevel(self._level)
                logger.addHandler(handler)
        return self
