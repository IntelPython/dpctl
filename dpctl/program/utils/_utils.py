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

"""Implements various utilities for the dpctl.program module."""

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
    OpDecorate = 71


class SpirvDecoration(IntEnum):
    SpecId = 1


def parse_spirv_specializations(
    spv_bytes: bytes | bytearray | memoryview,
) -> dict[int, dict[str, str]]:
    words = np.frombuffer(spv_bytes, dtype=np.uint32)

    # verify magic number
    if len(words) < 5 or words[0] != 0x07230203:
        raise ValueError("Invalid SPIR-V binary")

    types = {}
    ids = {}
    names = {}
    constants = {}

    i = 5  # skip 5 word header
    while i < len(words):
        word = words[i]
        opcode = word & 0xFFFF
        word_count = word >> 16

        if word_count == 0:
            raise ValueError(f"Invalid SPIR-V instruction at word index {i}")

        if opcode == SpirvOpCode.OpTypeBool:
            result_id = int(words[i + 1])
            types[result_id] = "?"
        elif opcode == SpirvOpCode.OpTypeInt:
            result_id = int(words[i + 1])
            width = int(words[i + 2])
            signed = int(words[i + 3])
            prefix = "i" if signed else "u"
            types[result_id] = f"{prefix}{width // 8}"
        elif opcode == SpirvOpCode.OpTypeFloat:
            result_id = int(words[i + 1])
            width = int(words[i + 2])
            types[result_id] = f"f{width // 8}"
        elif opcode in (
            SpirvOpCode.OpSpecConstant,
            SpirvOpCode.OpSpecConstantTrue,
            SpirvOpCode.OpSpecConstantFalse,
        ):
            type_id = int(words[i + 1])
            result_id = int(words[i + 2])
            constants[result_id] = type_id
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

    result = {}
    for target_id, spec_id in ids.items():
        type_id = constants.get(target_id)
        dtype_str = types.get(type_id, "unknown_type")
        name = names.get(target_id, f"unnamed_spec_const_{spec_id}")

        result[spec_id] = {
            "name": name,
            "dtype": dtype_str,
        }

    return result
