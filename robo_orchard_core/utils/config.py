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


"""Configuration class that extends Pydantic's model type."""

import ast
import importlib
import inspect
import io
import typing
from copy import deepcopy
from typing import Annotated, Any, Generic, Literal, Type

import rtoml as toml
import torch
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    GenerateSchema,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
)
from pydantic.functional_serializers import PlainSerializer, model_serializer
from pydantic.functional_validators import PlainValidator, model_validator
from pydantic_core import core_schema, from_json, to_json
from typing_extensions import Callable, ParamSpec, Self, TypeVar

from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.patches import patch_class_method
from robo_orchard_core.utils.registry import Registry

logger = LoggerManager().get_child(__name__)


T = TypeVar("T")

T_co = TypeVar("T_co", covariant=True)


T_contra = TypeVar("T_contra", contravariant=True)
V = TypeVar("V")
TYPE_LIST = ParamSpec("TYPE_LIST")

PYDANTIC_CONFIGCLASS = Registry("PYDANTIC_CONFIGCLASS")

TOML_NULL = "null"


@patch_class_method(GenerateSchema, "_unsubstituted_typevar_schema")
def _wrap_unsubstituted_typevar_schema(self, typevar: TypeVar):
    """Wraps the default `_unsubstituted_typevar_schema` method.

    This patch is used to support the serialization of TypeVar with default
    values that are the same as the bound type.

    """

    assert isinstance(typevar, typing.TypeVar)

    bound = typevar.__bound__

    try:
        typevar_has_default = typevar.has_default()  # type: ignore
    except AttributeError:
        typevar_has_default = getattr(typevar, "__default__", None) is not None

    if (
        typevar_has_default
        and bound is not None
        and (
            typing.get_origin(bound)
            == typing.get_origin(
                typevar.__default__,
            )
        )
    ):
        schema = self.generate_schema(bound)
        schema["serialization"] = (
            core_schema.wrap_serializer_function_ser_schema(
                lambda x, h: h(x), schema=core_schema.any_schema()
            )
        )
        return schema

    return self.__old__unsubstituted_typevar_schema(typevar)


def is_lambda_expression(name: str) -> bool:
    """Checks if the input string is a lambda expression.

    A copy of omni.isaac.lab.utils.config.is_lambda_expression.

    Args:
        name: The input string.

    Returns:
        Whether the input string is a lambda expression.
    """
    try:
        ast.parse(name)
        return isinstance(ast.parse(name).body[0], ast.Expr) and isinstance(
            ast.parse(name).body[0].value,  # type: ignore
            ast.Lambda,
        )
    except SyntaxError:
        return False


def callable_to_string(value: Callable) -> str:
    """Converts a callable object to a string.

    A copy of omni.isaac.lab.utils.config.callable_to_string.

    Note:
        This function only works for the following types of callable objects:
        - Class type
        - Function type, should be imported from a module.
        - Lambda function, should be defined in the same file. Once lambda
        function is deserialized from string, this function will not work!

    Args:
        value: A callable object.

    Raises:
        ValueError: When the input argument is not a callable object.

    Returns:
        str: A string representation of the callable object.

    """
    # check if callable

    if not callable(value):
        raise ValueError(f"The input argument is not callable: {value}.")
    # check if lambda function
    if value.__name__ == "<lambda>":
        return f"lambda {inspect.getsourcelines(value)[0][0].strip().split('lambda')[1].strip().split(',')[0]}"  # noqa
    else:
        # get the module and function name
        module_name = value.__module__
        function_name = value.__name__
        # handle nested class
        if isinstance(value, type):
            function_name = value.__qualname__
        # return the string
        return f"{module_name}:{function_name}"


def string_to_callable(name: str) -> Callable:
    """Resolves the module and function names to return the function.

    A copy of omni.isaac.lab.utils.config.string_to_callable.

    Args:
        name: The function name. The format should be 'module:attribute_name'
            or a lambda expression of format: 'lambda x: x'.

    Note:
        This function only works for the following types of callable objects:

        - Class type
        - Function type, should be imported from a module.
        - Lambda function, should be defined in the same file. Once lambda
          function is deserialized from string, this function will
          not work!


    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: When the module cannot be found.

    Returns:
        The function loaded from the module.

    """  # noqa: E501
    try:
        if is_lambda_expression(name):
            callable_object = eval(name)
        else:
            mod_name, attr_name = name.split(":")
            mod = importlib.import_module(mod_name)
            # handle nested class
            attr_names = attr_name.split(".")
            attr_name = attr_names[-1]
            if len(attr_names) > 1:
                for attr in attr_names[0:-1]:
                    mod = getattr(mod, attr)
            callable_object = getattr(mod, attr_name)
        # check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise AttributeError(
                f"The imported object is not callable: '{name}'"
            )
    except (ValueError, ModuleNotFoundError) as e:
        msg = (
            f"Could not resolve the input string '{name}' into callable object."  # noqa: E501
            " The format of input should be 'module:attribute_name'.\n"
            f"Received the error:\n {e}."
        )
        raise ValueError(msg)


_CallableSerializer = PlainSerializer(
    lambda x: (callable_to_string(x) if x is not None else None),
    return_type=str,
    when_used="always",
    # when_used="json",
)

ClassType_co = Annotated[
    type[T_co],
    PlainValidator(
        lambda x: string_to_callable(x) if isinstance(x, str) else x
    ),
    _CallableSerializer,
]

ClassType = ClassType_co


CallableType = Annotated[
    Callable[TYPE_LIST, T],
    PlainValidator(
        lambda x: string_to_callable(x) if isinstance(x, str) else x
    ),
    _CallableSerializer,
]

SliceType = Annotated[
    slice,
    PlainValidator(lambda x: slice(*x["slice"]) if isinstance(x, dict) else x),
    PlainSerializer(
        lambda x: {"slice": [x.start, x.stop, x.step]}, return_type=dict
    ),
]


TorchTensor = Annotated[
    torch.Tensor,
    PlainValidator(
        lambda x: torch.tensor(x) if not isinstance(x, torch.Tensor) else x
    ),
    PlainSerializer(lambda x: x.tolist(), return_type=list, when_used="json"),
]


class Config(BaseModel):
    """Base class for configuration classes."""

    __exclude_config_type__: bool = False
    """The flag to exclude the '__config_type__' key in the serialization."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    @model_serializer(mode="wrap", return_type=dict, when_used="always")
    def wrapped_model_ser(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ):
        """Serializes the configuration to a dictionary.

        This wrapper function is used when the configuration is serialized.
        It adds the `__config_type__` key to the dictionary.

        `__config_type__` is the string representation of the class type. It
        is used to determine the class type when deserializing the JSON string
        instead of using pydantic's default behavior.

        If a configuration class does not need the `__config_type__` key, set
        `__exclude_config_type__` to True in the configuration class.

        For builtin types, the `__config_type__` key will not be added to the
        dictionary.

        The `context` argument in the `model_dump` method is used to
        determine whether to include the `__config_type__` key in the
        serialized dictionary. If context['exclude_config_type'] is True,
        the `__config_type__` key will not be added to the dictionary.

        """
        if (
            (
                hasattr(self, "__exclude_config_type__")
                and self.__exclude_config_type__
            )
            or self.__class__.__module__ == "builtins"
            or (
                isinstance(info.context, dict)
                and info.context.get("exclude_config_type", True)
            )
        ):
            return handler(self)

        ret = {"__config_type__": callable_to_string(type(self))}
        ret.update(handler(self))
        return ret

    @model_validator(mode="wrap")
    @classmethod
    def wrapped_model_val(
        cls, data: Any, handler: ValidatorFunctionWrapHandler
    ):
        if isinstance(data, str):
            data = from_json(data, allow_partial=True)
        if isinstance(data, dict):
            if "__config_type__" in data:
                data = data.copy()
                target_cls = string_to_callable(data.pop("__config_type__"))
                if target_cls == cls:
                    return handler(data)
                else:
                    return target_cls.model_validate(data)
            else:
                return handler(data)
        return data

    def __post_init__(self):
        """Hack to replace __post_init__ in configclass.

        This hotfix is only needed for inheriting from both pydantic Config
        and omni.isaac.lab.utils.configclass.
        """
        pass

    def model_post_init(self, *args, **kwargs):
        """Post init method for the model.

        Perform additional initialization after __init__ and model_construct.
        This is useful if you want to do some validation that requires the
        entire model to be initialized.

        To be consistent with configclass, this method is implemented by
        calling the `__post_init__` method.

        """
        self.__post_init__()

    def to_dict(
        self,
        mode: Literal["python", "json"] = "python",
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        include_config_type: bool = False,
        **kwargs,
    ) -> dict:
        """Converts the configuration to a dictionary.

        This method will call pydanitc's `model_dump` method to convert the
        configuration to a dictionary. The `__config_type__` key will be added
        to the dictionary if `include_config_type` is True.

        Note:
            This method is not designed for serialization. Use the
            :py:meth:`to_str` method for serialization!

        Args:
            mode (Literal["python", "json"]): The mode of the output
                dictionary. If 'python', the output will be a Python
                dictionary. If 'json', the output will be a JSON serializable
                dictionary. Default is 'python'.
            exclude_unset (bool): Whether to exclude unset values from the
                dictionary. Default is False.
            exclude_defaults (bool): Whether to exclude default values from the
                dictionary. Default is False.
            exclude_none (bool): Whether to exclude None values from the
                dictionary. Default is False.
            include_config_type (bool): Whether to include the
                `__config_type__` key in the dictionary. If False, the
                deserialization will use the class type defined in the class
                annotation, not the acaual deserialized class type! This will
                break the consistency of serialization and deserialization.
                Default is False.

        """
        context = {
            "exclude_config_type": not include_config_type,
        }
        ret = self.model_dump(
            mode=mode,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            context=context,
            **kwargs,
        )
        return ret

    def to_str(
        self,
        format: Literal["json", "toml", "yaml"] = "json",
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        include_config_type: bool = True,
        **kwargs,
    ) -> str:
        """Converts the configuration to a string.

        Different from the `to_dict` method, this method adds the
        '__config_type__' key to the dictionary and converts the dictionary
        to a string by default.

        For config that does not need '__config_type__' key, set
        `__exclude_config_type__` to True in the config class or
        set `include_config_type` to False in the method call.

        Args:
            format (str): The format of the output string. Can be 'json',
                'yaml' or 'toml'. Default is 'json'.
            exclude_unset (bool): Whether to exclude unset values from the
                dictionary. Default is False.
            exclude_defaults (bool): Whether to exclude default values from the
                dictionary. Default is False.
            exclude_none (bool): Whether to exclude None values from the
                dictionary. Default is False.
            include_config_type (bool): Whether to include the
                `__config_type__` key in the string. If False, the
                deserialization will use the class type defined in the class
                annotation, not the actual deserialized class type! This will
                break the consistency of serialization and deserialization.
                Default is True.
            **kwargs: Additional keyword arguments to be passed to the
                serialization method :meth:`BaseModel.model_dump_json`.

        Returns:
            str: The string representation of the configuration.

        """

        json_str = self.model_dump_json(
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            context={
                "exclude_config_type": not include_config_type,
            },
            **kwargs,
        )

        if format == "json":
            return json_str
        elif format == "toml":
            data = from_json(json_str)
            return toml.dumps(data, none_value=TOML_NULL)
        elif format == "yaml":
            data = from_json(json_str)
            ret = yaml.dump(data, sort_keys=False)
            assert isinstance(ret, str)
            return ret
        else:
            raise ValueError(f"Unsupported format: {format}.")

    @classmethod
    def from_dict(cls: Type[Self], data: dict, **kwargs) -> Self:
        return cls.model_validate(data, **kwargs)

    @classmethod
    def from_str(
        cls: Type[Self],
        data: str,
        format: Literal["json", "toml", "yaml"] = "json",
        **kwargs,
    ) -> Self:
        """Creates a configuration object from a string.

        If the input string is not in JSON format, it will be converted to a
        JSON string before deserialization.

        Args:
            data (str): The input string data.
            format (str): The format of the input string. Can be 'json',
                'yaml' or 'toml'. Default is 'json'.
            **kwargs: Additional keyword arguments to be passed to the
                deserialization method :meth:`BaseModel.model_validate_json`.

        """
        if format == "json":
            return cls.model_validate_json(data, **kwargs)
        elif format == "toml":
            dict_data = toml.loads(data, none_value=TOML_NULL)
            json_str = to_json(dict_data).decode("utf-8")
            return cls.model_validate_json(json_str, **kwargs)
        elif format == "yaml":
            dict_data = yaml.load(io.StringIO(data), Loader=yaml.FullLoader)
            json_str = to_json(dict_data).decode("utf-8")
            return cls.model_validate_json(json_str, **kwargs)

    def copy(self) -> Self:
        """Returns a copy of the configuration."""
        return self.model_copy()

    def replace(self, **kwargs) -> Self:
        return self.model_copy(update=kwargs)

    def content_equal(self, other: Self) -> bool:
        """Check if the content of the configuration is equal to another.

        This method relies on the `to_json` method to convert the
        configuration to json dictionaries and compare them.
        """
        self_data = self.to_str(format="json")
        other_data = other.to_str(format="json")

        return self_data == other_data

    def __eq__(self, other: Any) -> bool:
        """Check if the configuration is equal to another object.

        This method checks if the other object is an instance of the same
        class and if the content of the configuration is equal to the other
        configuration.
        """
        if not isinstance(other, self.__class__):
            return False
        return self.content_equal(other)


class ClassConfig(Config, Generic[T_co]):
    """Configuration for a class type.

    This configuration class is used to store the initialization data
    for a class type.

    """

    class_type: ClassType_co[T_co]

    def __call__(self, *args, **kwargs) -> T_co:
        """Creates an instance of the class type with the configuration data.

        There are two ways to create an instance of the class type:
        1. By passing the keyword arguments from the configuration data to
            the class constructor
        2. By passing the configuration object to the class constructor

        If `class_type` is has the `InitFromConfig` attribute set to True,
        call `create_instance_by_cfg` method. Otherwise, call
        `create_instance_by_kwargs` method.
        """

        if getattr(self.class_type, "InitFromConfig", False):
            return self.create_instance_by_cfg(*args, **kwargs)
        else:
            return self.create_instance_by_kwargs(*args, **kwargs)

    def create_instance_by_kwargs(self, *args, **kwargs) -> T_co:
        """Creates an instance of the class type.

        This method is used to create an instance of the class type by
        passing the keyword arguments from the configuration data to the
        class constructor.

        The returned instance will be initialized with the configuration data.

        Args:
            *args: Additional positional arguments to be passed to the class
                constructor.
            **kwargs: Additional keyword arguments to be passed to the class
                constructor. These will override the configuration data.

        Returns:
            T: An instance of the class type.

        """
        dict_data = self.to_dict()
        dict_data.pop("class_type")
        if "__config_type__" in dict_data:
            dict_data.pop("__config_type__")

        dict_data.update(kwargs)
        return self.class_type(*args, **dict_data)

    def create_instance_by_cfg(self, *args, **kwargs) -> T_co:
        """Creates an instance of the class type.

        This method is used to create an instance of the class type by
        passing the configuration object, not the keyword arguments.
        Any positional arguments will be passed to the class constructor after
        the configuration object.

        Args:
            *args: Additional positional arguments to be passed to the class
                constructor after the configuration object.
            **kwargs: Additional keyword arguments to be passed to the class
                constructor. These will override the configuration data.

        Returns:
            T: An instance of the class type.

        """
        cfg = self.replace(**kwargs)
        return self.class_type(cfg, *args)  # type: ignore


class CallableConfig(Config, Generic[T_co]):
    """Configuration for a callable type.

    This configuration class is used to store parameters for a callable type.


    """

    func: CallableType[..., T_co]

    def __call__(self, **kwargs) -> T_co:
        """Calls the function with the configuration data.

        Args:
            **kwargs: Additional keyword arguments to be passed to the
                function. These will override the configuration data.

        """
        dict_data = self.to_dict()
        dict_data.pop("func")
        if "__config_type__" in dict_data:
            dict_data.pop("__config_type__")
        dict_data.update(kwargs)
        return self.func(**dict_data)


ConfigT = TypeVar("ConfigT", bound=Config)


def load_config_class(
    data: str | dict, format: Literal["json", "toml", "yaml"] = "json"
) -> Config:
    """Loads the configuration class from a JSON string or dictionary.

    Args:
        data (str | dict): The string data or dictionary.
        format (str): The format of the string input data. Can be 'json',
            'yaml' or 'toml'. Default is 'json'.
    """
    if isinstance(data, str):
        if format == "json":
            data = from_json(data, allow_partial=True)
        elif format == "toml":
            data = toml.loads(data, none_value=TOML_NULL)
        elif format == "yaml":
            data = yaml.load(io.StringIO(data), Loader=yaml.FullLoader)  # type: ignore
        else:
            raise ValueError(f"Unsupported format: {format}.")
    if isinstance(data, dict):
        if "__config_type__" in data:
            data = deepcopy(data)
            target_cls = string_to_callable(data.pop("__config_type__"))
            return target_cls.model_validate(data)
        else:
            raise ValueError(
                "The input data does not contain '__config_type__' key."
            )
    raise ValueError("The input data is not a dictionary or string.")


class ClassInitFromConfigMixin:
    """Mixin class for the configuration class that initializes from config."""

    InitFromConfig: bool = True
