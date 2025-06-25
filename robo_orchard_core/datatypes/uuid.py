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


import time
from os import urandom
from typing import Any, Optional, TypeAlias
from uuid import (
    UUID as _UUID,
    uuid1 as _uuid1,
)

from pydantic import (
    Field,
    model_serializer,
    model_validator,
)

from robo_orchard_core.datatypes.dataclass import DataClass


def _hex2bytes(hex: str) -> bytes:
    return bytes.fromhex(hex)


def _hex2int(hex: str) -> int:
    return int(hex, 16)


def _bytes2int(bytes: bytes, signed: bool = False) -> int:
    return int.from_bytes(bytes, "big", signed=signed)


int_ = int  # The built-in int type
bytes_ = bytes  # The built-in bytes type


class UUID64(DataClass):
    """A 64-bit UUID.

    The UUID64 is a 64-bit unsigned integer. It can be represented as a
    hex string, a bytes object or an int.

    Args:
        hex (str, optional): A 16-byte hex string.
        bytes (bytes, optional): A 8-byte bytes object.
        int (int, optional): A 64-bit integer.
        signed (bool, optional): Whether the int is signed. Defaults to False.

    """

    value: int = Field(
        title="UUID64 value",
        description="A 64-bit unsigned integer. Only for internal use.",
    )

    def __init__(
        self,
        hex: Optional[str] = None,
        bytes: Optional[bytes] = None,
        int: Optional[int] = None,
        signed: bool = False,
    ):
        if [hex, bytes, int].count(None) != 2:
            raise TypeError(
                "one of the hex, bytes or int arguments must be given"
            )
        value = None
        # self._int = None
        if hex is not None:
            # convert hex into bytes
            if len(hex) != 16:
                raise ValueError(
                    f"hex must be len(16) ! given: {hex} with len({len(hex)})"
                )
            # self._data = _hex2bytes(hex)
            value = _hex2int(hex)
        if bytes is not None:
            assert len(bytes) == 8, f"len(bytes)={len(bytes)}"
            value = _bytes2int(bytes)
        if int is not None:
            if signed:
                assert -(2**63) <= int < 2**63
                # store the signed int as an unsigned int
                value = _bytes2int(int_(int).to_bytes(8, "big", signed=True))
            else:
                assert 0 <= int < 2**64
                value = int

        assert value is not None
        super().__init__(value=value)

    @model_serializer
    def serialize(self) -> str:
        return self.hex

    @model_validator(mode="before")
    @classmethod
    def validate(cls, data: Any):
        if isinstance(data, str):
            if len(data) != 16:
                raise ValueError(
                    f"hex must be len(16) ! given: {data} with len({len(data)})"  # noqa: E501
                )
            value = _hex2int(data)
            return {"value": value}
        return data

    @property
    def hex(self) -> str:
        return self.int.to_bytes(8, "big").hex()

    @property
    def int(self) -> int:
        return self.value  # type: ignore

    @property
    def signed_int(self) -> int:  # type: ignore
        return _bytes2int(self.int.to_bytes(8, "big"), signed=True)

    @property
    def bytes(self) -> bytes:
        return self.int.to_bytes(8, "big")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UUID64):
            return self.int == other.int
        else:
            return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, UUID64):
            return self.int < other.int
        else:
            return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, UUID64):
            return self.int > other.int
        else:
            return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, UUID64):
            return self.int <= other.int
        else:
            return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, UUID64):
            return self.int >= other.int
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.int)

    def __repr__(self) -> str:
        return f"UUID64('{self.hex}')"

    def __str__(self) -> str:
        return self.__repr__()


UUID128: TypeAlias = _UUID


def uuid64() -> UUID64:
    """Generate a UUID64 that use sequential patterns.

    Algorithm:
        1. Get the current time in microseconds.
        2. Get 12 bits of random bytes.
        3. Concatenate the random bytes to the end of the current time.

    Returns:
        UUID64:  A UUID64 object.
    """
    return UUID64(
        hex=int(time.time() * 1000000).to_bytes(8, "big").hex()[3:]
        + urandom(2).hex()[0:3]
    )


def uuid128() -> UUID128:
    """Generate a UUID128 that use sequential patterns.

    The UUID128 generated by this function is suitable for primary key.


    Algorithm:
        1. Generate UUID using uuid.uuid1().
        2. Swap the time_low and time_hi_version fields.

    Returns:
        UUID128:  A UUID128 object.
    """
    hex_data = _uuid1().hex
    reversed_hex_data = (
        hex_data[12:16] + hex_data[8:12] + hex_data[0:8] + hex_data[16:]
    )
    return UUID128(hex=reversed_hex_data)


def as_hex32(hex_value: str) -> str:
    """Convert a hex string to a 32 byte hex string.

    If the input hex string is less than 32 bytes, it will be padded with 0s.
    If the input hex string is more than 32 bytes, exception will be raised.

    Args:
        hex_value (str): A hex string.

    Returns:
        str: A 32-bit hex string.
    """
    if len(hex_value) > 32:
        raise ValueError("hex_value is too long")

    # check if value is valid hex
    try:
        int(hex_value, 16)
    except ValueError:
        raise ValueError("hex_value is not a valid hex string")

    return hex_value.zfill(32)
