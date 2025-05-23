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
#
# -----------------------------------------------------------------------------
# Portions of this file are derived from Isaac Lab (https://github.com/isaac-sim/IsaacLab).
# The original Isaac Lab code is licensed under the BSD-3-Clause license.
#
# Original Isaac Lab Copyright Notice:
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You can find the original Isaac Lab source code at:
# https://github.com/isaac-sim/IsaacLab
#
# Modifications to the derived code, if any, are Copyright (c) 2025
# Horizon Robotics and are licensed under the Apache License, Version 2.0.
# The combined work in this file is distributed under the Apache License,
# Version 2.0, subject to the conditions of the BSD-3-Clause license for the
# portions derived from Isaac Lab.

import re
from collections.abc import Sequence
from typing import Any, List, Tuple


def resolve_matching_names(
    keys: str | Sequence[str],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
) -> Tuple[List[int], List[str]]:
    """Match a list of query regular expressions against a list of strings and return the matched indices and names.

    When a list of query regular expressions is provided, the function checks each target string against each
    query regular expression and returns the indices of the matched strings and the matched strings.

    If the :attr:`preserve_order` is True, the ordering of the matched indices and names is the same as the order
    of the provided list of strings. This means that the ordering is dictated by the order of the target strings
    and not the order of the query regular expressions.

    If the :attr:`preserve_order` is False, the ordering of the matched indices and names is the same as the order
    of the provided list of query regular expressions.

    For example, consider the list of strings is ['a', 'b', 'c', 'd', 'e'] and the regular expressions are ['a|c', 'b'].
    If :attr:`preserve_order` is False, then the function will return the indices of the matched strings and the
    strings as: ([0, 1, 2], ['a', 'b', 'c']). When :attr:`preserve_order` is True, it will return them as:
    ([0, 2, 1], ['a', 'c', 'b']).

    Note:
        The function does not sort the indices. It returns the indices in the order they are found.

    Args:
        keys: A regular expression or a list of regular expressions to match the strings in the list.
        list_of_strings: A list of strings to match.
        preserve_order: Whether to preserve the order of the query keys in the returned values. Defaults to False.

    Returns:
        A tuple of lists containing the matched indices and names.

    Raises:
        ValueError: When multiple matches are found for a string in the list.
        ValueError: When not all regular expressions are matched.

    Reference:
        https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/utils/string.py
    """  # noqa: E501
    # resolve name keys
    if isinstance(keys, str):
        keys = [keys]
    # find matching patterns
    index_list = []
    names_list = []
    key_idx_list = []
    # book-keeping to check that we always have a one-to-one mapping
    # i.e. each target string should match only one regular expression
    target_strings_match_found = [None for _ in range(len(list_of_strings))]
    keys_match_found = [[] for _ in range(len(keys))]
    # loop over all target strings
    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, re_key in enumerate(keys):
            if re.fullmatch(re_key, potential_match_string):
                # check if match already found
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"  # noqa: E501
                    )
                # add to list
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                key_idx_list.append(key_index)
                # add for regex key
                keys_match_found[key_index].append(potential_match_string)
    # reorder keys if they should be returned in order of the query keys
    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(keys)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1
        # reorder index and names list
        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
        # update
        index_list = index_list_reorder
        names_list = names_list_reorder
    # check that all regular expressions are matched
    if not all(keys_match_found):
        # make this print nicely aligned for debugging
        msg = "\n"
        for key, value in zip(keys, keys_match_found, strict=False):
            msg += f"\t{key}: {value}\n"
        msg += f"Available strings: {list_of_strings}\n"
        # raise error
        raise ValueError(
            f"Not all regular expressions are matched! Please check that the regular expressions are correct: {msg}"  # noqa: E501
        )
    # return
    return index_list, names_list  # type: ignore


def resolve_matching_names_values(
    data: dict[str, Any],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
) -> tuple[list[int], list[str], list[Any]]:
    """Match a list of regular expressions in a dictionary. against a list of strings and return the matched indices, names, and values.

    If the :attr:`preserve_order` is True, the ordering of the matched indices and names is the same as the order
    of the provided list of strings. This means that the ordering is dictated by the order of the target strings
    and not the order of the query regular expressions.

    If the :attr:`preserve_order` is False, the ordering of the matched indices and names is the same as the order
    of the provided list of query regular expressions.

    For example, consider the dictionary is {"a|d|e": 1, "b|c": 2}, the list of strings is ['a', 'b', 'c', 'd', 'e'].
    If :attr:`preserve_order` is False, then the function will return the indices of the matched strings, the
    matched strings, and the values as: ([0, 1, 2, 3, 4], ['a', 'b', 'c', 'd', 'e'], [1, 2, 2, 1, 1]). When
    :attr:`preserve_order` is True, it will return them as: ([0, 3, 4, 1, 2], ['a', 'd', 'e', 'b', 'c'], [1, 1, 1, 2, 2]).

    Args:
        data: A dictionary of regular expressions and values to match the strings in the list.
        list_of_strings: A list of strings to match.
        preserve_order: Whether to preserve the order of the query keys in the returned values. Defaults to False.

    Returns:
        A tuple of lists containing the matched indices, names, and values.

    Raises:
        TypeError: When the input argument :attr:`data` is not a dictionary.
        ValueError: When multiple matches are found for a string in the dictionary.
        ValueError: When not all regular expressions in the data keys are matched.

    Reference:
        https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/utils/string.py
    """  # noqa: E501
    # check valid input
    if not isinstance(data, dict):
        raise TypeError(
            f"Input argument `data` should be a dictionary. Received: {data}"
        )
    # find matching patterns
    index_list = []
    names_list = []
    values_list = []
    key_idx_list = []
    # book-keeping to check that we always have a one-to-one mapping
    # i.e. each target string should match only one regular expression
    target_strings_match_found: list[str | None] = [
        None for _ in range(len(list_of_strings))
    ]
    keys_match_found = [[] for _ in range(len(data))]
    # loop over all target strings
    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, (re_key, value) in enumerate(data.items()):
            if re.fullmatch(re_key, potential_match_string):
                # check if match already found
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"  # noqa: E501
                    )
                # add to list
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                values_list.append(value)
                key_idx_list.append(key_index)
                # add for regex key
                keys_match_found[key_index].append(potential_match_string)
    # reorder keys if they should be returned in order of the query keys
    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(data)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1
        # reorder index and names list
        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        values_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
            values_list_reorder[reorder_idx] = values_list[idx]
        # update
        index_list = index_list_reorder
        names_list = names_list_reorder
        values_list = values_list_reorder
    # check that all regular expressions are matched
    if not all(keys_match_found):
        # make this print nicely aligned for debugging
        msg = "\n"
        for key, value in zip(data.keys(), keys_match_found, strict=False):
            msg += f"\t{key}: {value}\n"
        msg += f"Available strings: {list_of_strings}\n"
        # raise error
        raise ValueError(
            "Not all regular expressions are matched! Please check that "
            f"the regular expressions are correct: {msg}"
        )
    # return
    return index_list, names_list, values_list


def add_indentation(
    text: str,
    indent: int = 4,
    indent_char: str = " ",
    first_line_indent: bool = True,
) -> str:
    """Add indentation to a string.

    Args:
        text (str): The input string to be indented. The string may contain
            multiple lines, and the indentation will be applied to each line.
        indent (int, optional): The number of spaces to use for indentation.
            Defaults to 4.
        indent_char (str, optional): The character to use for indentation.
            Defaults to " ".
        first_line_indent (bool, optional): Whether to indent the first line.
            Defaults to True.

    Returns:
        str: The indented string.
    """
    # Create the indentation string
    indent_str = indent_char * indent
    # Split the text into lines
    lines = text.splitlines()
    # Add indentation to each line
    indented_lines = []
    for i, line in enumerate(lines):
        if i == 0 and not first_line_indent:
            indented_lines.append(line)
        else:
            indented_lines.append(indent_str + line)
    # Join the lines back together
    ret = "\n".join(indented_lines)
    if text.endswith("\n"):
        ret += "\n"
    return ret
