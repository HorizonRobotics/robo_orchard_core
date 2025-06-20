# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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
import warnings
from typing import Callable


class ExperimentalWarning(UserWarning):
    """Custom warning for experimental features.

    This warning is used to indicate that a feature is experimental
    and may be subject to breaking changes in the future.
    It inherits from `UserWarning` to allow for easy filtering or
    handling by users.
    """

    pass


def experimental(fn: Callable) -> Callable:
    """Decorator to mark a function as experimental.

    Applying this decorator will issue a warning when the function is called,
    indicating that the function is experimental and may change in the future.


    Example:
    .. code-block:: python

        from robo_orchard_core.utils.experimental import experimental


        @experimental
        def my_experimental_function():
            pass

    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        warnings.warn(
            (
                f"'{fn.__name__}' is experimental and might be changed in the future."  # noqa: E501
            ),
            ExperimentalWarning,
        )
        return fn(*args, **kwargs)

    return wrapped
