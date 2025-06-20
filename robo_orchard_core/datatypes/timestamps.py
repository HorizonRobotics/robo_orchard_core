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

from typing import Literal


def concat_timestamps(
    data: list[list[int] | None], concat_dim: Literal["row", "col"] = "row"
) -> list[int] | None:
    """Concatenate a batch of timestamps.

    Args:
        data (list[list[int]|None]): A batch of timestamps, each element
            is a list of timestamps or None.
        concat_dim (Literal["row", "col"], optional): The dimension to
            concatenate along. If "row", timestamps are concatenated
            along the first dimension (batch dimension). If "col", timestamps
            are checked for consistency across the batch to be the same.
            Defaults to "row".

    Returns:
        list[int]|None: Concatenated timestamps or None if all
            elements are None.
    """
    if len(data) == 0:
        return None

    cnt_none = data.count(None)

    if cnt_none == len(data):
        return None

    if cnt_none > 0 and cnt_none < len(data):
        raise ValueError(
            "Timestamps must be either all None or all set. "
            f"Found {cnt_none} None and {len(data) - cnt_none} set."
        )

    # Now we know all timestamps are set

    if concat_dim == "row":
        return [ts for sublist in data for ts in sublist]  # type: ignore
    elif concat_dim == "col":
        # Check if all timestamps are the same
        first_ts = data[0]
        if any(ts != first_ts for ts in data):
            raise ValueError(
                "Timestamps must match across all elements when concatenating "
                "along the column dimension."
            )
        return first_ts
