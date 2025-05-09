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


"""Helper functions for patching classes, methods, and functions."""

import functools
import warnings


def patch_class_method(
    cls, method_name: str, check_method_name_exists: bool = False
):
    """A decorator to patch a class method.

    The old implementation of the method is renamed to `__old_{method_name}`.

    If the method is already patched, a warning is issued and the method is
    not patched again.

    Args:
        cls (type): The class to patch.
        method_name (str): The method name to patch.
        check_method_name_exists (bool): Whether to check if the method
            already exists in the class. If True, an error will be raised if
            the method does not exist. Default: False.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        old_impl_name = f"__old_{method_name}"
        if hasattr(cls, old_impl_name):
            warnings.warn(
                f"Method {method_name} in class {cls.__name__} is already "
                "patched. Skipping."
            )
            return wrapper

        old_impl = getattr(cls, method_name, None)

        if check_method_name_exists and old_impl is None:
            raise ValueError(
                f"Method {method_name} does not exists in class {cls.__name__}"
            )

        if old_impl is not None:
            setattr(cls, f"__old_{method_name}", old_impl)

        setattr(cls, method_name, wrapper)

        return wrapper

    return decorator
