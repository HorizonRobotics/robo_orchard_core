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

"""Base data class and mixin."""

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from robo_orchard_core.utils.torch_utils import Device, make_device

__all__ = ["DataClass", "TensorToMixin", "tensor_equal"]


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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def __eq__(self, other: Any) -> bool:
        # use the default equality method first.
        try:
            return super().__eq__(other)
        except Exception:
            pass

        def obj_eq(src, dst):
            """Custom equality method for objects with tensor support."""
            if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
                return tensor_equal(src, dst)
            elif isinstance(src, (list, tuple)) and isinstance(
                dst, (list, tuple)
            ):
                if len(src) != len(dst):
                    return False
                return all(obj_eq(s, d) for s, d in zip(src, dst, strict=True))
            elif isinstance(src, dict) and isinstance(dst, dict):
                if src.keys() != dst.keys():
                    return False
                return all(obj_eq(src[k], dst[k]) for k in src.keys())
            else:
                return src == dst

        # if the default equality method fails, we use the custom
        # equality method that compares the __dict__ attributes of the objects.
        if not isinstance(other, DataClass):
            raise NotImplementedError()
        return obj_eq(self.__dict__, other.__dict__)


class TensorToMixin:
    def to(
        self,
        device: Device,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> Self:
        """Move or cast the tensors/modules in the data class.

        This method performs in-place conversion of all tensors and modules
        in the data class to the specified device and dtype.

        Args:
            device (Device): The target device to move the tensors/modules to.
            dtype (torch.dtype | None, optional): The target dtype to cast
                the tensors to. If None, the dtype will not be changed.
            non_blocking (bool, optional): If True, the operation will be
                performed in a non-blocking manner. Defaults to False.

        """

        def apply_to(obj, device, dtype, non_blocking):
            if isinstance(obj, torch.Tensor):
                return obj.to(
                    device=device, dtype=dtype, non_blocking=non_blocking
                )
            elif isinstance(obj, torch.nn.Module):
                return obj.to(
                    device=device, dtype=dtype, non_blocking=non_blocking
                )
            elif isinstance(obj, list):
                return [
                    apply_to(
                        item,
                        device=device,
                        dtype=dtype,
                        non_blocking=non_blocking,
                    )
                    for item in obj
                ]
            elif isinstance(obj, tuple):
                return tuple(
                    apply_to(
                        item,
                        device=device,
                        dtype=dtype,
                        non_blocking=non_blocking,
                    )
                    for item in obj
                )
            elif isinstance(obj, dict):
                return {
                    k: apply_to(
                        v,
                        device=device,
                        dtype=dtype,
                        non_blocking=non_blocking,
                    )
                    for k, v in obj.items()
                }
            else:
                return obj

        device = make_device(device)
        for k, obj in self.__dict__.items():
            self.__dict__[k] = apply_to(
                obj, device=device, dtype=dtype, non_blocking=non_blocking
            )

        return self


def tensor_equal(
    src: torch.Tensor | None, dst: torch.Tensor | None, atol: float = 1e-8
) -> bool:
    """Check if two tensors are equal within a tolerance.

    Args:
        src (torch.Tensor | None): The first tensor.
        dst (torch.Tensor | None): The second tensor.
        eps (float, optional): The tolerance for equality. Defaults to 1e-6.

    Returns:
        bool: True if the tensors are equal within the tolerance,
            False otherwise.
    """
    if [src, dst].count(None) == 1:
        return False
    if src is None and dst is None:
        return True
    assert src is not None and dst is not None
    return torch.allclose(src, dst, atol)
