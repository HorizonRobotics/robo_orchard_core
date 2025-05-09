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

from collections import OrderedDict
from typing import Any


def flatten_dict(
    d: dict | list | tuple,
    parent_key: str = "",
    sep: str = "/",
    keep_order: bool = False,
) -> dict[str, Any]:
    """Flatten a nested dictionary or list into a flat dictionary.

    For example, given the following dictionary:

    {
        "a": 1,
        "b": {
            "c": 2,
            "d": [3, 4],
        }
    }

    The flattened dictionary would be:

    {
        "a": 1,
        "b/c": 2,
        "b/d/index_0": 3,
        "b/d/index_1": 4,
    }


    Args:
        d (dict|list|tuple): The dictionary or list to flatten.
        parent_key (str): The parent key of the dictionary.
        sep (str): The separator to use between keys.
        keep_order (bool): Whether to keep the order of the items.

    Returns:
        dict: The flattened dictionary.
    """

    items = []
    if isinstance(d, dict):
        item_iter = d.items()
    elif isinstance(d, (list, tuple)):
        item_iter = enumerate(d)

    for k, v in item_iter:
        if isinstance(k, int):
            k = f"index_{k}"
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, list, tuple)):
            items.extend(
                flatten_dict(
                    v, new_key, sep=sep, keep_order=keep_order
                ).items()
            )
        else:
            items.append((new_key, v))
    if keep_order:
        return OrderedDict(items)
    return dict(items)
