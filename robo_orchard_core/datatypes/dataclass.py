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

"""Base data class that extends pydantic's BaseModel."""

from pydantic import BaseModel

__all__ = ["DataClass"]


class DataClass(BaseModel):
    """The base data class that extends pydantic's BaseModel.

    This class is used to define data classes that are used to store data
    and validate the data. It extends pydantic's BaseModel and adds a
    :py:meth:`__post_init__` method that can be used to perform additional
    initialization after the model is constructed.

    Note:
        Serialization and deserialization using pydantic's methods are not
        recommended for performance reasons, as data classes can be used to
        store large tensors or other data that are not easily serialized.

        User should implement the proper serialization and deserialization
        methods when needed.

    """

    def __post_init__(self):
        """Hack to replace __post_init__ in configclass."""
        pass

    def model_post_init(self, *args, **kwargs):
        """Post init method for the model.

        Perform additional initialization after :py:meth:`__init__`
        and model_construct. This is useful if you want to do some validation
        that requires the entire model to be initialized.

        To be consistent with configclass, this method is implemented by
        calling the :py:meth:`__post_init__` method.

        """
        self.__post_init__()
