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

import pytest

from robo_orchard_core.datatypes.adaptor import (
    TypeAdaptorFactory,
    TypeAdaptorFactoryConfig,
    TypeAdaptorImpl,
    TypeAdaptorImplConfig,
)


class Int2StrAdaptor(TypeAdaptorImpl[int, str]):
    def __init__(self, cfg: TypeAdaptorImplConfig[int, str] | None = None):
        super().__init__()
        if cfg is None:
            cfg = TypeAdaptorImplConfig(class_type=Int2StrAdaptor)
        self._cfg = cfg

    def __call__(self, value):
        return str(value)


class List2StrAdaptor(TypeAdaptorImpl[list, str]):
    def __init__(self, cfg: TypeAdaptorImplConfig[list, str] | None = None):
        super().__init__()
        if cfg is None:
            cfg = TypeAdaptorImplConfig(class_type=List2StrAdaptor)

        self._cfg = cfg

    def __call__(self, value):
        return str(value)


class TestTypeAdaptor:
    def test_type_adaptor_impl(self):
        adaptor = Int2StrAdaptor()
        assert adaptor(1) == "1"
        adaptor = List2StrAdaptor()
        assert adaptor([1, 2, 3]) == "[1, 2, 3]"

    def test_type_adaptor_source_type(self):
        adaptor = Int2StrAdaptor()
        assert adaptor.source_type is int
        assert adaptor.target_type is str

        adaptor = List2StrAdaptor()
        assert adaptor.source_type is list
        assert adaptor.target_type is str

    def test_type_adaptor_factory(self):
        factory_cfg = TypeAdaptorFactoryConfig(
            class_type=TypeAdaptorFactory,
            adaptors=[
                TypeAdaptorImplConfig(class_type=Int2StrAdaptor),
                TypeAdaptorImplConfig(class_type=List2StrAdaptor),
            ],
        )

        factory = TypeAdaptorFactory(factory_cfg)
        assert factory(1) == "1"
        assert factory([1, 2, 3]) == "[1, 2, 3]"

    def test_type_adaptor_factory_source_type(self):
        factory_cfg = TypeAdaptorFactoryConfig(
            class_type=TypeAdaptorFactory,
            adaptors=[
                TypeAdaptorImplConfig(class_type=Int2StrAdaptor),
                TypeAdaptorImplConfig(class_type=List2StrAdaptor),
            ],
        )

        factory = TypeAdaptorFactory(factory_cfg)
        assert factory.source_types == set([int, list])

    def test_register_adaptor(self):
        factory = TypeAdaptorFactory(TypeAdaptorFactoryConfig())
        factory.register(Int2StrAdaptor())
        factory.register(List2StrAdaptor())
        assert factory(1) == "1"
        assert factory([1, 2, 3]) == "[1, 2, 3]"

    def test_register_duplicate_adaptor(self):
        factory = TypeAdaptorFactory()
        factory.register(Int2StrAdaptor())
        with pytest.raises(ValueError):
            factory.register(Int2StrAdaptor())
