#                      Data Parallel Control (dpctl)
#
# Copyright 2020 Intel Corporation
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
#
# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

""" Implements SyclPlatform Cython extension type.
"""

from ._backend cimport (  # noqa: E211
    DPCTLCString_Delete,
    DPCTLDeviceSelector_Delete,
    DPCTLFilterSelector_Create,
    DPCTLPlatform_Copy,
    DPCTLPlatform_Create,
    DPCTLPlatform_CreateFromSelector,
    DPCTLPlatform_Delete,
    DPCTLPlatform_GetBackend,
    DPCTLPlatform_GetName,
    DPCTLPlatform_GetPlatforms,
    DPCTLPlatform_GetVendor,
    DPCTLPlatform_GetVersion,
    DPCTLPlatformMgr_GetInfo,
    DPCTLPlatformMgr_PrintInfo,
    DPCTLPlatformVector_Delete,
    DPCTLPlatformVector_GetAt,
    DPCTLPlatformVector_Size,
    DPCTLPlatformVectorRef,
    DPCTLSyclDeviceSelectorRef,
    DPCTLSyclPlatformRef,
    _backend_type,
)

import warnings

from .enum_types import backend_type

__all__ = [
    "get_platforms",
    "lsplatform",
    "SyclPlatform",
]

cdef class _SyclPlatform:
    """ Data owner for SyclPlatform
    """

    def __dealloc__(self):
        DPCTLPlatform_Delete(self._platform_ref)
        DPCTLCString_Delete(self._name)
        DPCTLCString_Delete(self._vendor)
        DPCTLCString_Delete(self._version)


cdef void _init_helper(_SyclPlatform platform, DPCTLSyclPlatformRef PRef):
    "Populate attributes of class from opaque reference PRef"
    platform._platform_ref = PRef
    platform._name = DPCTLPlatform_GetName(PRef)
    platform._version = DPCTLPlatform_GetVersion(PRef)
    platform._vendor = DPCTLPlatform_GetVendor(PRef)


cdef class SyclPlatform(_SyclPlatform):
    """ SyclPlatform(self, arg=None)
        Python class representing ``cl::sycl::platform`` class.

        SyclPlatform() - create platform selected by sycl::default_selector
        SyclPlatform(filter_selector) - create platform selected by filter
        selector
    """
    @staticmethod
    cdef SyclPlatform _create(DPCTLSyclPlatformRef pref):
        """
        This function calls ``DPCTLPlatform_Delete(pref)``.

        The user of this function must pass a copy to keep the
        pref argument alive.
        """
        cdef _SyclPlatform p = _SyclPlatform.__new__(_SyclPlatform)
        # Initialize the attributes of the SyclPlatform object
        _init_helper(<_SyclPlatform>p, pref)
        return SyclPlatform(p)

    cdef int _init_from__SyclPlatform(self, _SyclPlatform other):
        self._platform_ref = DPCTLPlatform_Copy(other._platform_ref)
        if (self._platform_ref is NULL):
            return -1
        self._name = DPCTLPlatform_GetName(self._platform_ref)
        self._version = DPCTLPlatform_GetVersion(self._platform_ref)
        self._vendor = DPCTLPlatform_GetVendor(self._platform_ref)

    cdef int _init_from_cstring(self, const char *string):
        cdef DPCTLSyclDeviceSelectorRef DSRef = NULL
        DSRef = DPCTLFilterSelector_Create(string)
        ret = self._init_from_selector(DSRef)
        return ret

    cdef int _init_from_selector(self, DPCTLSyclDeviceSelectorRef DSRef):
        # Initialize the SyclPlatform from a DPCTLSyclDeviceSelectorRef
        cdef DPCTLSyclPlatformRef PRef = DPCTLPlatform_CreateFromSelector(DSRef)
        DPCTLDeviceSelector_Delete(DSRef)
        if PRef is NULL:
            return -1
        else:
            _init_helper(self, PRef)
            return 0

    cdef DPCTLSyclPlatformRef get_platform_ref(self):
        """ Returns the ``DPCTLSyclPlatformRef`` pointer for this class.
        """
        return self._platform_ref

    @property
    def __name__(self):
        return "SyclPlatform"

    def __repr__(self):
        return (
            "<dpctl."
            + self.__name__
            + " ["
            + self.name
            + ", "
            + self.vendor
            + ", "
            + " "
            + self.version + "] at {}>".format(hex(id(self)))
        )

    def __cinit__(self, arg=None):
        if type(arg) is unicode:
            string = bytes(<unicode>arg, "utf-8")
            filter_c_str = string
            ret = self._init_from_cstring(filter_c_str)
            if ret == -1:
                raise ValueError(
                    "Could not create a SyclPlatform with the selector string"
                )
        elif isinstance(arg, unicode):
            string = bytes(<unicode>unicode(arg), "utf-8")
            filter_c_str = string
            ret = self._init_from_cstring(filter_c_str)
            if ret == -1:
                raise ValueError(
                    "Could not create a SyclPlatform with the selector string"
                )
        elif isinstance(arg, _SyclPlatform):
            ret = self._init_from__SyclPlatform(arg)
            if ret == -1:
                raise ValueError(
                    "Could not create SyclPlatform from _SyclPlatform instance"
                )
        elif arg is None:
            PRef = DPCTLPlatform_Create()
            if PRef is NULL:
                raise ValueError(
                    "Could not create a SyclPlatform from default selector"
                )
            else:
                _init_helper(self, PRef)
        else:
            raise ValueError(
                "Invalid argument. Argument should be a str object specifying "
                "a SYCL filter selector string."
            )

    def print_platform_info(self, verbosity=0):
        """ Print information about the SYCL platform.

        The level of information printed out by the function can be controlled
        by the optional ``vebosity`` setting.

            - **Verbosity level 0**: Prints out the list of platforms and their
              names.
            - **Verbosity level 1**: Prints out the name, version, vendor,
              backend, number of devices for each platform.
            - **Verbosity level 2**: At the highest level of verbosity
              everything in the previous levels along with the name, version,
              and filter string for each device is printed.

        Args:
            verbosity (optional): Defaults to 0.
                The verbosity controls how much information is printed by the
                function. 0 is the lowest level set by default and 2 is the
                highest level to print the most verbose output.

        """
        cdef size_t v = 0

        if not isinstance(verbosity, int):
            warnings.warn(
                "Illegal verbosity level. Accepted values are 0, 1, or 2. "
                "Using the default verbosity level of 0."
            )
        else:
            v = <size_t>(verbosity)
            if v > 2:
                warnings.warn(
                    "Illegal verbosity level. Accepted values are 0, 1, or 2. "
                    "Using the default verbosity level of 0."
                )
                v = 0
        DPCTLPlatformMgr_PrintInfo(self._platform_ref, v)

    @property
    def vendor(self):
        """ Returns the platform vendor name as a string.
        """
        return self._vendor.decode()

    @property
    def version(self):
        """ Returns a backend-defined driver version as a string.
        """
        return self._version.decode()

    @property
    def name(self):
        """ Returns the name of the platform as a string
        """
        return self._name.decode()

    @property
    def backend(self):
        """Returns the backend_type enum value for this device

        Returns:
            backend_type: The backend for the device.
        """
        cdef _backend_type BTy = (
            DPCTLPlatform_GetBackend(self._platform_ref)
        )
        if BTy == _backend_type._CUDA:
            return backend_type.cuda
        elif BTy == _backend_type._HOST:
            return backend_type.host
        elif BTy == _backend_type._LEVEL_ZERO:
            return backend_type.level_zero
        elif BTy == _backend_type._OPENCL:
            return backend_type.opencl
        else:
            raise ValueError("Unknown backend type.")


def lsplatform(verbosity=0):
    """
    Prints out the list of available SYCL platforms, and optionally extra
    metadata about each platform.

    The level of information printed out by the function can be controlled by
    the optional ``vebosity`` setting.

    - **Verbosity level 0**: Prints out the list of platforms and their names.
    - **Verbosity level 1**: Prints out the name, version, vendor, backend,
      number of devices for each platform.
    - **Verbosity level 2**: At the highest level of verbosity everything in the
      previous levels along with the name, version, and filter string for each
      device is printed.

    At verbosity level 2 (highest level) the following output is generated.

    :Example:
        On a system with an OpenCL CPU driver, OpenCL GPU driver,
        Level Zero GPU driver, running the command. ::

        $python -c "import dpctl; dpctl.lsplatform()"

        returns ::

            Platform 0::
                Name        Intel(R) OpenCL
                Version     OpenCL 2.1 LINUX
                Vendor      Intel(R) Corporation
                Profile     FULL_PROFILE
                Backend     opencl
                Devices     1
                    Device 0::
                    Name            Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
                    Driver version  2020.11.11.0.13_160000
                    Device type     cpu
            Platform 1::
                Name        Intel(R) OpenCL HD Graphics
                Version     OpenCL 3.0
                Vendor      Intel(R) Corporation
                Profile     FULL_PROFILE
                Backend     opencl
                Devices     1
                    Device 0::
                    Name            Intel(R) Graphics Gen9 [0x3e98]
                    Driver version  20.47.18513
                    Device type     gpu
            Platform 2::
                Name        Intel(R) Level-Zero
                Version     1.0
                Vendor      Intel(R) Corporation
                Profile     FULL_PROFILE
                Backend     level_zero
                Devices     1
                    Device 0::
                    Name            Intel(R) Graphics Gen9 [0x3e98]
                    Driver version  1.0.18513
                    Device type     gpu

    Args:
        verbosity (optional): Defaults to 0.
            The verbosity controls how much information is printed by the
            function. 0 is the lowest level set by default and 2 is the highest
            level to print the most verbose output.
    """
    cdef DPCTLPlatformVectorRef PVRef = NULL
    cdef size_t v = 0
    cdef size_t size = 0
    cdef const char * info_str = NULL
    cdef DPCTLSyclPlatformRef PRef = NULL

    if not isinstance(verbosity, int):
        print(
            "Illegal verbosity level. Accepted values are 0, 1, or 2. "
            "Using the default verbosity level of 0."
        )
    else:
        v = <size_t>(verbosity)
        if v > 2:
            warnings.warn(
                "Illegal verbosity level. Accepted values are 0, 1, or 2. "
                "Using the default verbosity level of 0."
            )
            v = 0

    PVRef = DPCTLPlatform_GetPlatforms()

    if PVRef is not NULL:
        size = DPCTLPlatformVector_Size(PVRef)
        for i in range(size):
            if v != 0:
                print("Platform ", i, "::")
            PRef = DPCTLPlatformVector_GetAt(PVRef, i)
            info_str = DPCTLPlatformMgr_GetInfo(PRef,v)
            py_info = <bytes> info_str
            DPCTLCString_Delete(info_str)
            DPCTLPlatform_Delete(PRef)
            print(py_info.decode("utf-8"),end='')
    DPCTLPlatformVector_Delete(PVRef)


cpdef list get_platforms():
    """
    Returns a list of all available SYCL platforms on the system.

    Returns:
        list: A list of SYCL platforms on the system.
    """
    cdef list platforms = []
    cdef DPCTLPlatformVectorRef PVRef = NULL
    cdef size_t size = 0

    PVRef = DPCTLPlatform_GetPlatforms()
    if PVRef is not NULL:
        size = DPCTLPlatformVector_Size(PVRef)
        for i in range(size):
            PRef = DPCTLPlatformVector_GetAt(PVRef, i)
            P = SyclPlatform._create(PRef)
            platforms.append(P)

    DPCTLPlatformVector_Delete(PVRef)
    return platforms
