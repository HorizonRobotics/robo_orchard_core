# ruff: noqa: E501 D415 D205

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


"""Best Practices for Config
================================================
"""

##################################################################
# Most applications require a configuration file to define the behavior,
# parameters, and settings of the application.  In this document, we will
# discuss the best practices for designing a configuration framework.
#
#
# Guiding Principles
# -------------------
#
# When you deal with a complex system, there are many aspects to consider
# when designing the configuration framework:
#
# * How is it passed to the application? (e.g., command-line arguments,
#   environment variables, or configuration files)
# * Is the configuration compatible with IDEs? (e.g., code completion)
# * How to validate the configuration?
# * How to provide default values?
# * Is the configuration file human-readable?
#
# Based on those considerations, we think a good configuration framework should
# have the following features:
#
# * **IDE-friendly**: The configuration should be compatible with IDEs, such as
#   VSCode. It should support code completion and type hints. This is the most
#   important feature for users.
#
# * **Validation**: The configuration should be validated before it is used.
#   Validation includes static type checking, name checking, and value checking
#   via IDEs, and runtime checking. The validation should provide helpful error
#   messages, and should be checked as early as possible.
#
# * **Serialization**: The configuration should be serializable to a
#   human-readable format, such as JSON, YAML, or TOML. This is important for
#   persisting the configuration to disk and for sharing.
#
#
# `Registration` pattern is widely used in many frameworks. It is a good way to
# provide a mapping between a string and a class. However, it is not a good fit
# for configuration design. The **abuse of `Registration` pattern** can lead
# to **difficult-to-understand code** and make it **hard to find the
# relationship between the configuration and the code**.
#
#
# Config Class Design
# -------------------
#
# Config class is a data class that contains all the configuration parameters.
# We implement the :py:class:`~robo_orchard_core.utils.config.Config` class
# as a subclass of :py:class:`pydantic.BaseModel`, providing methods for
# serialization, deserialization, and validation. Pydantic is a data validation
# library, which provides all the features we need for configuration.
#
# To use the Config class, you need to define a subclass of Config and define
# the fields you need. The Config class will automatically generate the
# serialization and deserialization methods for you.
#

# %%
# Simple Example
# ^^^^^^^^^^^^^^
# Here is a simple example of a Config class:

from __future__ import annotations

from pydantic import ValidationError

from robo_orchard_core.utils.config import Config

# sphinx_gallery_thumbnail_path = '_static/images/sphx_glr_install_thumb.png'


class MyConfig(Config):
    """My Config class."""

    id: int
    name: str


cfg1 = MyConfig(id=1, name="test")
print("cfg1: ", cfg1)

# %%
# If you pass the wrong type, Config will raise a ValidationError:

# Wrong type
try:
    cfg2 = MyConfig(id="id1", name="test")  # type: ignore
except ValidationError as e:
    print(e)

# %%
# Wrong field is not allowed in Config:

try:
    cfg2 = MyConfig(id=1, name_new="test")  # type: ignore
except ValidationError as e:
    print(e)

# %%
# For more advanced usage or examples, please refer to the
# `Pydantic documentation <https://docs.pydantic.dev/latest/>`_.
#
#

# %%
# Config Serialization and Deserialization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Config class provides serialization and deserialization methods. You can
# serialize the Config object to a dictionary, json, yaml, or toml format.
# You can also deserialize the Config object.
#
# You can use the :py:meth:`~robo_orchard_core.utils.config.Config.to_dict`
# method to serialize the Config object to a dictionary:

cfg_dict = cfg1.to_dict()
print("cfg_dict: ", cfg_dict)

# %%
# To serialize the Config object to string format, you can use the
# :py:meth:`~robo_orchard_core.utils.config.Config.to_str` method:

# Serialize to json format. You can also serialize to yaml or toml format.
cfg1_json: str = cfg1.to_str(format="json")
print("json: \n", cfg1_json)


# %%
# You may notice that the serialized json string contains the type information
# of the config class: **`__config_type__`**, which is not a part of the config
# class definition. This is a reserved field name used for deserialization to
# determine the actual config class type, and only exists in the serialized
# string!
#
# To deserialize the Config object from string format, you can use the
# :py:meth:`~robo_orchard_core.utils.config.Config.from_str` method:
import os  # noqa: E402

from robo_orchard_core.utils.config import load_config_class  # noqa: E402

cfg2 = load_config_class(cfg1_json)

print("cfg2: ", cfg2)

# %%
# Handling Function and Class Type
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Sometimes you may need to pass a callback function or class type to the
# configuration. To handle this, one solution is to use `pickle` module
# to serialize the function or class. However, this approach will generate
# binary data, which is not human-readable. Binary data is not suitable for
# configuration files.
#
# To solve this problem, we serialize the function or class type to a string
# with the `module_name.class_name` format. This format is human-readable
# and can be easily deserialized. Here is an example to demonstrate how to
# handle callbacks and class types in the configuration:

from typing import Any  # noqa: E402

from robo_orchard_core.utils.config import (  # noqa: E402
    CallableType,
    ClassType,
)


def callback_func():
    print("callback function")


class ConfigWithCallback(Config):
    id: int
    name: str
    class_type: ClassType[Any]
    callback: CallableType


cfg3 = ConfigWithCallback(
    id=1, name="test", class_type=MyConfig, callback=callback_func
)

print(cfg3)
print(cfg3.to_str())

# %%
# .. note::
#
#     Handling function and class type in `module_name.class_name` format does
#     **not guarantee** that the function or class is available in the current
#     environment. Users need to ensure that the function or class **can be
#     imported** in the current environment, and the function or class is
#     **compatible with the current environment**.


# %%
# Class Config as Class Initializer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Some classes require multiple parameters to initialize. It is not convenient
# to pass all the parameters as arguments. In this case, you can use the Config
# class as the initialization parameter to simplify the initialization process.
#
# In this way, field of parameters are statically defined, and the class
# initialization is more readable and IDE-friendly:

from robo_orchard_core.utils.config import ClassConfig  # noqa: E402


class MyClass:
    def __init__(self, cfg: MyClassConfig):
        self.cfg = cfg

    def __repr__(self):
        return f"MyClass(cfg={self.cfg})"


class MyClassConfig(ClassConfig[MyClass]):
    class_type: ClassType[MyClass] = MyClass

    id: int
    name: str

    def __call__(self):
        return self.class_type(cfg=self)


cls_cfg = MyClassConfig(id=1, name="test")
my_cls = MyClass(cfg=cls_cfg)

print(my_cls)


# %%
# A ClassConfig can also be used to create a class instance:

my_cls2 = cls_cfg()
print(my_cls2)


# %%
# If we combine the ClassConfig with deserialization, we can easily create a
# class instance from a serialized string or a config file:

# from string
cls_cfg_json = cls_cfg.to_str(format="json")
cls_cfg2 = load_config_class(cls_cfg_json)
my_cls3 = cls_cfg2()
print(my_cls3)

# from config file
with open("cls_cfg.json", "w") as f:
    f.write(cls_cfg_json)

cls_cfg3 = load_config_class(open("cls_cfg.json").read())
my_cls4 = cls_cfg3()

print(my_cls4)
# clean up
os.remove("cls_cfg.json")


# %%
# Summary
# -------
# In this document, we discussed the best practices for designing a
# configuration framework. The Config class is one of the basic components
# in RoboOrchard, and is widely used in many modules.  Hope this document
# helps you understand the design of the Config class and how to use it.
