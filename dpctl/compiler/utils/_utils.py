#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements various utilities for the dpctl.compiler module."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class SpirvOpCode(IntEnum):
    OpName = 5
    OpTypeBool = 20
    OpTypeInt = 21
    OpTypeFloat = 22
    OpSpecConstantTrue = 48
    OpSpecConstantFalse = 49
    OpSpecConstant = 50
    OpFunction = 54
    OpDecorate = 71


class SpirvDecoration(IntEnum):
    SpecId = 1


@dataclass(frozen=True)
class SpecializationConstantInfo:
    """Data class representing specialization constant information."""

    spec_id: int
    dtype: str
    name: str
    itemsize: int
    default_value: int | float | bool | None


def parse_spirv_specializations(
    spv_bytes: bytes | bytearray | memoryview,
) -> tuple[SpecializationConstantInfo]:
    """
    Parses SPIR-V byte stream to extract information about specializations,
    including the specialization IDs, types, names, and default values.

    Note that the dtype information may be imprecise, as the compiler may
    choose to, for example, represent a bool as char, or may represent both
    signed and unsigned integers as unsigned integer bit buckets of the same
    length.

    Args:
        spv_bytes (bytes | bytearray | memoryview):
            the SPIR-V byte stream.

    Returns:
        tuple[SpecializationConstantInfo]:
            a tuple of parsed constants and their information represented by
            `SpecializationConstantInfo` objects, sorted by their
            specialization IDs. The length of the tuple is equal to the number
            of specialization constants found. Each
            `SpecializationConstantInfo` object contains the following
            attributes:

            - `spec_id` (int): The specialization ID.
            - `dtype` (str): A NumPy style string representing the data type.
            - `itemsize` (int): The size of the specialization constant in
                bytes.
            - `name` (str): The variable name. If not preserved in the binary,
                a default name in the format `unnamed_spec_const_{spec_id}` is
                used.
            - `default_value` (int | float | bool | None): The default value of
                the specialization constant. If not specified, `None` is used.
    """
    words = np.frombuffer(spv_bytes, dtype=np.uint32)

    # verify magic number
    if len(words) < 5 or words[0] != 0x07230203:
        raise ValueError("Invalid SPIR-V binary")

    types = {}
    ids = {}
    names = {}
    constants = {}
    defaults = {}

    i = 5  # skip 5 word header
    while i < len(words):
        word = words[i]
        opcode = word & 0xFFFF
        word_count = word >> 16

        if word_count == 0:
            raise ValueError(f"Invalid SPIR-V instruction at word index {i}")

        if opcode == SpirvOpCode.OpFunction:
            # everything following is not relevant to specialization constant
            # parsing, so we can stop parsing at this point
            break
        elif opcode == SpirvOpCode.OpTypeBool:
            result_id = int(words[i + 1])
            types[result_id] = {"dtype": "?", "itemsize": 1}
        elif opcode == SpirvOpCode.OpTypeInt:
            result_id = int(words[i + 1])
            width = int(words[i + 2])
            signed = int(words[i + 3])
            prefix = "i" if signed else "u"
            types[result_id] = {
                "dtype": f"{prefix}{width // 8}",
                "itemsize": width // 8,
            }
        elif opcode == SpirvOpCode.OpTypeFloat:
            result_id = int(words[i + 1])
            width = int(words[i + 2])
            types[result_id] = {
                "dtype": f"f{width // 8}",
                "itemsize": width // 8,
            }
        elif opcode == SpirvOpCode.OpSpecConstant:
            type_id = int(words[i + 1])
            result_id = int(words[i + 2])
            constants[result_id] = type_id
            literal_words = words[i + 3 : i + word_count]
            defaults[result_id] = literal_words.tobytes()
        elif opcode == SpirvOpCode.OpSpecConstantTrue:
            type_id = int(words[i + 1])
            result_id = int(words[i + 2])
            constants[result_id] = type_id
            defaults[result_id] = True
        elif opcode == SpirvOpCode.OpSpecConstantFalse:
            type_id = int(words[i + 1])
            result_id = int(words[i + 2])
            constants[result_id] = type_id
            defaults[result_id] = False
        elif opcode == SpirvOpCode.OpDecorate:
            target_id = int(words[i + 1])
            decoration = int(words[i + 2])
            if decoration == SpirvDecoration.SpecId:
                ids[target_id] = int(words[i + 3])
        elif opcode == SpirvOpCode.OpName:
            target_id = int(words[i + 1])
            name_bytes = words[i + 2 : i + word_count].tobytes()
            names[target_id] = name_bytes.split(b"\x00", 1)[0].decode("utf-8")

        i += word_count

    # a spec ID may appear multiple times in the same binary with different
    # target IDs. We only need to keep one, so skip duplicates
    unique_ids = set()
    result = []
    for target_id, spec_id in ids.items():
        if spec_id in unique_ids:
            continue
        unique_ids.add(spec_id)
        type_id = constants.get(target_id)
        type_info = types.get(type_id, {"dtype": "unknown_type", "itemsize": 0})
        name = names.get(target_id, f"unnamed_spec_const_{spec_id}")

        dtype_str = type_info["dtype"]
        raw_default = defaults.get(target_id)
        default_value = None
        if isinstance(raw_default, bytes):
            try:
                default_value = np.frombuffer(raw_default, dtype=dtype_str)[
                    0
                ].item()
            except Exception:
                default_value = None

        result.append(
            SpecializationConstantInfo(
                spec_id=spec_id,
                dtype=dtype_str,
                name=name,
                itemsize=type_info["itemsize"],
                default_value=default_value,
            )
        )

    return tuple(sorted(result, key=lambda x: x.spec_id))
