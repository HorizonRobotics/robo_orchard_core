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

from __future__ import annotations
import argparse
from typing import Optional, TypeVar

from pydantic_settings import BaseSettings, CliApp, CliSettingsSource

from robo_orchard_core.utils.config import Config


class SettingConfig(Config, BaseSettings):
    __exclude_config_type__: bool = True


PydanticBaseSettings = TypeVar("PydanticBaseSettings", bound=BaseSettings)


def pydantic_from_argparse(
    cls: type[PydanticBaseSettings],
    parser: argparse.ArgumentParser,
    cli_enforce_required: Optional[bool] = True,
    cli_avoid_json: Optional[bool] = True,
    **kwargs,
) -> PydanticBaseSettings:
    """Parse command line arguments using Pydantic and return the settings object.

    Args:
        cls (type[PydanticBaseSettings]): The Pydantic settings class to parse.
        parser (argparse.ArgumentParser): The argument parser to use.
        cli_enforce_required (Optional[bool], optional): Whether to enforce required
            arguments. Manually set to None if you want to leave it to the
            default behavior of CliSettingsSource. Defaults to True.
        cli_avoid_json (Optional[bool], optional): Whether to avoid JSON format.
            Manually set to None if you want to leave it to the
            default behavior of CliSettingsSource. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the
            CliSettingsSource constructor.

    Returns:
        PydanticBaseSettings: The parsed settings object.

    """  # noqa: E501

    cli_settings = CliSettingsSource(
        cls,
        root_parser=parser,
        cli_enforce_required=cli_enforce_required,
        cli_avoid_json=cli_avoid_json,
        **kwargs,
    )
    try:
        return CliApp.run(cls, cli_settings_source=cli_settings)
    except Exception as e:
        parser.error(str(e))
