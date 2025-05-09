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

import functools
import logging
import time
from typing import Callable, Optional, Tuple, Type

from robo_orchard_core.utils.logging import LoggerManager

logger = LoggerManager().get_child(__name__)

RETRY_ERROR_TYPES = Type[Exception] | Tuple[Type[Exception], ...]


class SingletonMixin:
    """A singleton mixin class."""

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance


def auto_retry(
    error_type: RETRY_ERROR_TYPES = Exception,
    error_callback: Optional[Callable[[Exception], None]] = None,
    retry_times: int = 3,
    delay: float = 0.1,
    logger: logging.Logger = logger,
):
    """A decorator to auto retry a function.

    Args:
        error_type (Type[Exception] | Tuple[Type[Exception], ...]): The error
            type to catch. Default: Exception.
        error_callback (Optional[Callable[[Exception], None]]): The error
            callback function. Default: None.
        retry_times (int): The retry times. Default: 3.
        delay (float): The delay time. Default: 0.1.
        logger (logging.Logger): The logger. Default: logger.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if isinstance(func, functools.partial):
                func_name = func.func
            else:
                func_name = func
            for i in range(retry_times):
                try:
                    return func(*args, **kwargs)
                except error_type as e:
                    if i < retry_times - 1:
                        logger.warning(
                            f"Auto retry {i + 1}/{retry_times} for function "
                            f"{func_name} due to {e}"
                        )
                        if error_callback is not None:
                            error_callback(e)
                        time.sleep(delay)
                    else:
                        if error_callback is not None:
                            error_callback(e)
                        raise e

        return wrapper

    return decorator
