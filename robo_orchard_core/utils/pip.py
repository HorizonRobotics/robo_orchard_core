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
import re
import subprocess
import sys


@functools.lru_cache(maxsize=None)
def get_package_version(package_name: str) -> str | None:
    """Get the package version.

    Args:
        package_name (str): The package name.

    Returns:
        str: The package version.
    """
    try:
        reqs = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", package_name]
        )
    except subprocess.CalledProcessError:
        return None
    reqs = reqs.decode("utf-8")
    # use regex to get the version number
    reg_find_res = re.search("Version: (.+)", reqs)
    if reg_find_res is None:
        raise ValueError(
            f"Cannot get the version of package {package_name}."
            f"pip show result: {reqs}"
        )

    version = reg_find_res.group(1)
    return version
