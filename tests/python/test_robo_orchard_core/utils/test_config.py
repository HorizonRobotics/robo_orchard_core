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

import pytest

from robo_orchard_core.utils.config import (
    CallableConfig,
    CallableType,
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType,
    Config,
)


def f1() -> int:
    return 1


def plus_10(v: int) -> int:
    return v + 10


def dummy_decorator_with_wraps(func):
    @functools.wraps(func)
    def wrapper():
        return func() + 10

    return wrapper


def dummy_decorator(func):
    def wrapper():
        return func() + 20

    return wrapper


@dummy_decorator
def f1_decorated() -> int:
    return 1


@dummy_decorator_with_wraps
def f1_decorated_with_wraps() -> int:
    return 1


class DummyConfig(Config):
    int_value: int = 100


class DummyClassConfig(DummyConfig, ClassConfig[DummyConfig]):
    class_type: ClassType[DummyConfig] = DummyConfig


class DummyConfig2(Config):
    cfg1: DummyClassConfig
    cfg2: DummyClassConfig


class DummyClassConfig2(DummyConfig2, ClassConfig[DummyConfig2]):
    class_type: ClassType[DummyConfig2] = DummyConfig2


class DummyCallableConfig(CallableConfig[int]):
    func: CallableType = f1


class Plus10CallableConfig(CallableConfig[int]):
    func: CallableType = plus_10
    v: int = 10


class DummyConfigInitializerMeta(ClassInitFromConfigMixin):
    def __init__(self, cfg: "DummyConfigInitializerMetaCfg"):
        self.cfg = cfg
        self.init_value = cfg.int_value

    def __str__(self):
        return f"DummyConfigInitializerMeta({self.init_value})"


class DummyConfigInitializerMetaCfg(
    DummyConfig, ClassConfig[DummyConfigInitializerMeta]
):
    class_type: ClassType[DummyConfigInitializerMeta] = (
        DummyConfigInitializerMeta
    )


class TestSimpleConfig:
    def test_simple_config(self):
        config = DummyConfig()
        assert config.int_value == 100
        config.int_value = 200
        assert config.int_value == 200

    def test_json_dump(self):
        config = DummyConfig()
        config.int_value = 200
        assert (
            config.to_str(format="json")
            == '{"__config_type__":"test_config:DummyConfig","int_value":200}'
        )

    def test_from_json(self):
        config = DummyConfig.from_str('{"int_value":200}', format="json")
        assert config.int_value == 200

    def test_dict_dump(self):
        config = DummyConfig()
        config.int_value = 200
        assert config.to_dict() == {
            "int_value": 200,
        }

    def test_dict_dump_with_type(self):
        config = DummyConfig()
        config.int_value = 200
        assert config.to_dict(include_config_type=True) == {
            "__config_type__": "test_config:DummyConfig",
            "int_value": 200,
        }

    def test_from_dict(self):
        config = DummyConfig.from_dict({"int_value": 200})
        assert config.int_value == 200


class TestSimpleClassConfig:
    def test_to_dict_python(self):
        config = DummyClassConfig()
        config.int_value = 200
        assert config.to_dict() == {
            "class_type": "test_config:DummyConfig",
            "int_value": 200,
        }

    def test_class_config_json_serialization(self):
        config = DummyClassConfig()
        config.int_value = 200
        assert callable(config.class_type)
        json_str = config.to_str(format="json")

        new_config = DummyClassConfig.from_str(json_str, format="json")
        assert new_config.int_value == config.int_value
        assert new_config.class_type == config.class_type

        # make sure that class_type is still callable
        assert callable(new_config.class_type)

    def test_class_config_create_instance(self):
        config = DummyClassConfig()
        config.int_value = 200
        instance = config()
        assert instance.int_value == 200
        assert isinstance(instance, DummyConfig)

    def test_class_config_create_instance_with_override(self):
        config = DummyClassConfig()
        config.int_value = 200
        instance = config(int_value=300)
        assert instance.int_value == 300
        assert isinstance(instance, DummyConfig)

    def test_class_config_call_with_ConfigAsArgInitMeta(self):
        config = DummyConfigInitializerMetaCfg()
        config.int_value = 200
        instance = config()
        assert instance.init_value == 200
        assert isinstance(instance, DummyConfigInitializerMeta)


class TestCallableConfig:
    def test_callable_config(self):
        config = DummyCallableConfig()
        assert config.func() == 1

    def test_callable_config_json_serialization(self):
        config = DummyCallableConfig()
        json_str = config.to_str(format="json")
        new_config = DummyCallableConfig.from_str(json_str, format="json")
        assert new_config.func() == 1

    def test_callable_config_call(self):
        config = Plus10CallableConfig()
        assert config() == 20

    def test_callable_config_call_with_override(self):
        config = Plus10CallableConfig()
        assert config(v=20) == 30

    def test_callable_config_call_with_decorator(self):
        config = DummyCallableConfig()
        config.func = f1_decorated
        assert config() == 21

    def test_callable_config_call_with_decorator_with_wraps(self):
        config = DummyCallableConfig()
        config.func = f1_decorated_with_wraps
        assert config() == 11


class TestCascadeClassConfig:
    def test_cascade_class_config(self):
        config = DummyClassConfig2(
            cfg1=DummyClassConfig(int_value=200),
            cfg2=DummyClassConfig(int_value=300),
        )
        assert config.to_dict() == {
            "class_type": "test_config:DummyConfig2",
            "cfg1": {
                "class_type": "test_config:DummyConfig",
                "int_value": 200,
            },
            "cfg2": {
                "class_type": "test_config:DummyConfig",
                "int_value": 300,
            },
        }

    def test_cascade_class_config_json_serialization(self):
        config = DummyClassConfig2(
            cfg1=DummyClassConfig(int_value=200),
            cfg2=DummyClassConfig(int_value=300),
        )
        json_str = config.to_str(format="json")
        new_config = DummyClassConfig2.from_str(json_str, format="json")
        assert new_config.to_dict() == config.to_dict()

    def test_cascade_class_config_create_instance(self):
        config = DummyClassConfig2(
            cfg1=DummyClassConfig(int_value=200),
            cfg2=DummyClassConfig(int_value=300),
        )
        instance = config.create_instance_by_kwargs()
        assert isinstance(instance, DummyConfig2)
        assert instance.to_dict() == {
            "cfg1": {
                "class_type": "test_config:DummyConfig",
                "int_value": 200,
            },
            "cfg2": {
                "class_type": "test_config:DummyConfig",
                "int_value": 300,
            },
        }


if __name__ == "__main__":
    pytest.main(["-s", "test_config.py"])
