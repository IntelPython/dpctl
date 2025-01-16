#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np

_npver = np.lib.NumpyVersion(np.__version__)

if _npver < "1.25.0":  # pragma: no cover
    from numpy import AxisError
else:
    from numpy.exceptions import AxisError

if _npver >= "2.0.0":
    from numpy._core.numeric import normalize_axis_index, normalize_axis_tuple
else:  # pragma: no cover
    from numpy.core.numeric import normalize_axis_index, normalize_axis_tuple


__all__ = ["AxisError", "normalize_axis_index", "normalize_axis_tuple"]
