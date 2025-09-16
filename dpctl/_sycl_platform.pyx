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
#
# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# cython: freethreading_compatible = True

""" Implements SyclPlatform Cython extension type.
"""

from libcpp cimport bool

from ._backend cimport (  # noqa: E211
    DPCTLCString_Delete,
    DPCTLDeviceSelector_Delete,
    DPCTLDeviceVector_Delete,
    DPCTLDeviceVector_GetAt,
    DPCTLDeviceVector_Size,
    DPCTLDeviceVectorRef,
    DPCTLFilterSelector_Create,
    DPCTLPlatform_AreEq,
    DPCTLPlatform_Copy,
    DPCTLPlatform_Create,
    DPCTLPlatform_CreateFromSelector,
    DPCTLPlatform_Delete,
    DPCTLPlatform_GetBackend,
    DPCTLPlatform_GetCompositeDevices,
    DPCTLPlatform_GetDefaultContext,
    DPCTLPlatform_GetDevices,
    DPCTLPlatform_GetName,
    DPCTLPlatform_GetPlatforms,
    DPCTLPlatform_GetVendor,
    DPCTLPlatform_GetVersion,
    DPCTLPlatform_Hash,
    DPCTLPlatformMgr_GetInfo,
    DPCTLPlatformMgr_PrintInfo,
    DPCTLPlatformVector_Delete,
    DPCTLPlatformVector_GetAt,
    DPCTLPlatformVector_Size,
    DPCTLPlatformVectorRef,
    DPCTLSyclContextRef,
    DPCTLSyclDeviceRef,
    DPCTLSyclDeviceSelectorRef,
    DPCTLSyclPlatformRef,
    _backend_type,
    _device_type,
)

import warnings

from ._sycl_context import SyclContextCreationError
from .enum_types import backend_type
from .enum_types import device_type as device_type_t

from ._sycl_context cimport SyclContext
from ._sycl_device cimport SyclDevice

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
    Python class representing ``sycl::platform`` class.

    There are two ways of creating a :class:`.SyclPlatform`
    instance:

    - Invoking the constructor with no arguments creates a
      platform using the default selector.

    :Example:
        .. code-block:: python

            import dpctl

            # Create a SyclPlatform for default-selected device
            pl = dpctl.SyclPlatform()
            print(pl.name, pl.version)

    - Invoking the constructor with specific filter selector string that
      creates a queue for the device corresponding to the filter string.

    :Example:
        .. code-block:: python

            import dpctl

            # Create a SyclPlatform for device selected by
            # filter-selector string
            pl = dpctl.SyclPlatform("opencl:cpu")
            print(pl.name, pl.version)
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
            verbosity (Literal[0, 1, 2], optional):.
                The verbosity controls how much information is printed by the
                function. Value ``0`` is the lowest level set by default and
                ``2`` is the highest level to print the most verbose output.
                Default: ``0``
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
        """
        Returns the platform vendor name as a string.

        Returns:
            str:
                Vendor name
        """
        return self._vendor.decode()

    @property
    def version(self):
        """ Returns a backend-defined driver version as a string.

        Returns:
            str:
                Version of the backend-defined driver
        """
        return self._version.decode()

    @property
    def name(self):
        """ Returns the name of the platform as a string.

        Returns:
            str:
                Name of the platform
        """
        return self._name.decode()

    @property
    def backend(self):
        """Returns the backend_type enum value for this platform

        Returns:
            backend_type:
                The backend for the platform.
        """
        cdef _backend_type BTy = (
            DPCTLPlatform_GetBackend(self._platform_ref)
        )
        if BTy == _backend_type._CUDA:
            return backend_type.cuda
        elif BTy == _backend_type._HIP:
            return backend_type.hip
        elif BTy == _backend_type._LEVEL_ZERO:
            return backend_type.level_zero
        elif BTy == _backend_type._OPENCL:
            return backend_type.opencl
        else:
            raise ValueError("Unknown backend type.")

    @property
    def default_context(self):
        """Returns the default platform context for this platform

        Returns:
            :class:`dpctl.SyclContext`
                The default context for the platform.
        Raises:
            SyclContextCreationError
                If default_context is not supported
        """
        cdef DPCTLSyclContextRef CRef = (
            DPCTLPlatform_GetDefaultContext(self._platform_ref)
        )

        if (CRef == NULL):
            raise SyclContextCreationError(
                "Getting default_context ran into a problem"
            )
        else:
            return SyclContext._create(CRef)

    cdef bool equals(self, SyclPlatform other):
        """
        Returns true if the :class:`dpctl.SyclPlatform` argument has the
        same underlying ``DPCTLSyclPlatformRef`` object as this
        :class:`dpctl.SyclPlatform` instance.

        Returns:
            bool:
                ``True`` if the two :class:`dpctl.SyclPlatform` objects
                point to the same ``DPCTLSyclPlatformRef`` object, otherwise
                ``False``.
        """
        return DPCTLPlatform_AreEq(self._platform_ref, other.get_platform_ref())

    def __eq__(self, other):
        """
        Returns True if the :class:`dpctl.SyclPlatform` argument has the
        same underlying ``DPCTLSyclPlatformRef`` object as this
        :class:`dpctl.SyclPlatform` instance.

        Returns:
            bool:
                ``True`` if the two :class:`dpctl.SyclPlatform` objects
                point to the same ``DPCTLSyclPlatformRef`` object, otherwise
                ``False``.
        """
        if isinstance(other, SyclPlatform):
            return self.equals(<SyclPlatform> other)
        else:
            return False

    def __hash__(self):
        """
        Returns a hash value by hashing the underlying ``sycl::platform``
        object.

        Returns:
            int:
                Hash value
        """
        return DPCTLPlatform_Hash(self._platform_ref)

    def get_devices(self, device_type=device_type_t.all):
        """
        Returns the list of :class:`dpctl.SyclDevice` objects associated with
        :class:`dpctl.SyclPlatform` instance selected based on
        the given :class:`dpctl.device_type`.

        Args:
            device_type (str, :class:`dpctl.device_type`, optional):
                A :class:`dpctl.device_type` enum value or a string that
                specifies a SYCL device type. Currently, accepted values are:
                "gpu", "cpu", "accelerator", or "all", and their equivalent
                ``dpctl.device_type`` enumerators.
                Default: ``dpctl.device_type.all``.

        Returns:
            list:
                A :obj:`list` of :class:`dpctl.SyclDevice` objects
                that belong to this platform.

        Raises:
            TypeError:
                If `device_type` is not a string or :class:`dpctl.device_type`
                enum.
            ValueError:
                If the ``DPCTLPlatform_GetDevices`` call returned
                ``NULL`` instead of a ``DPCTLDeviceVectorRef`` object.
        """
        cdef _device_type DTy = _device_type._ALL_DEVICES
        cdef DPCTLDeviceVectorRef DVRef = NULL
        cdef size_t num_devs
        cdef size_t i
        cdef DPCTLSyclDeviceRef DRef

        if isinstance(device_type, str):
            dty_str = device_type.strip().lower()
            if dty_str == "accelerator":
                DTy = _device_type._ACCELERATOR
            elif dty_str == "all":
                DTy = _device_type._ALL_DEVICES
            elif dty_str == "cpu":
                DTy = _device_type._CPU
            elif dty_str == "gpu":
                DTy = _device_type._GPU
            else:
                DTy = _device_type._UNKNOWN_DEVICE
        elif isinstance(device_type, device_type_t):
            if device_type == device_type_t.all:
                DTy = _device_type._ALL_DEVICES
            elif device_type == device_type_t.accelerator:
                DTy = _device_type._ACCELERATOR
            elif device_type == device_type_t.cpu:
                DTy = _device_type._CPU
            elif device_type == device_type_t.gpu:
                DTy = _device_type._GPU
            else:
                DTy = _device_type._UNKNOWN_DEVICE
        else:
            raise TypeError(
                "device type should be specified as a str or an "
                "``enum_types.device_type``."
            )
        DVRef = DPCTLPlatform_GetDevices(self.get_platform_ref(), DTy)
        if (DVRef is NULL):
            raise ValueError("Internal error: NULL device vector encountered")
        num_devs = DPCTLDeviceVector_Size(DVRef)
        devices = []
        for i in range(num_devs):
            DRef = DPCTLDeviceVector_GetAt(DVRef, i)
            devices.append(SyclDevice._create(DRef))
        DPCTLDeviceVector_Delete(DVRef)

        return devices

    def get_composite_devices(self):
        """
        Returns the list of composite :class:`dpctl.SyclDevice` objects
        associated with :class:`dpctl.SyclPlatform` instance.

        Returns:
            list:
                A :obj:`list` of composite :class:`dpctl.SyclDevice` objects
                that belong to this platform.

        Raises:
            ValueError:
                If the ``DPCTLPlatform_GetCompositeDevices`` call returned
                ``NULL`` instead of a ``DPCTLDeviceVectorRef`` object.
        """
        cdef DPCTLDeviceVectorRef DVRef = NULL
        cdef size_t num_devs
        cdef size_t i
        cdef DPCTLSyclDeviceRef DRef

        DVRef = DPCTLPlatform_GetCompositeDevices(self.get_platform_ref())
        if (DVRef is NULL):
            raise ValueError("Internal error: NULL device vector encountered")
        num_devs = DPCTLDeviceVector_Size(DVRef)
        composite_devices = []
        for i in range(num_devs):
            DRef = DPCTLDeviceVector_GetAt(DVRef, i)
            composite_devices.append(SyclDevice._create(DRef))
        DPCTLDeviceVector_Delete(DVRef)

        return composite_devices


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
        Level Zero GPU driver, running the command:

        .. code-block:: bash

            $ python -c "import dpctl; dpctl.lsplatform(verbosity=2)"

        outputs

        .. code-block:: text
            :caption: Sample output of lsplatform(verbosity=2)

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
        verbosity (Literal[0,1,2], optional):
            The verbosity controls how much information is printed by the
            function. 0 is the lowest level set by default and 2 is the highest
            level to print the most verbose output. Default: `0`.
    """
    cdef DPCTLPlatformVectorRef PVRef = NULL
    cdef size_t v = 0
    cdef size_t size = 0
    cdef const char * info_str = NULL
    cdef DPCTLSyclPlatformRef PRef = NULL

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

    PVRef = DPCTLPlatform_GetPlatforms()

    if PVRef is not NULL:
        size = DPCTLPlatformVector_Size(PVRef)
        for i in range(size):
            if v != 0:
                print("Platform ", i, "::")
            PRef = DPCTLPlatformVector_GetAt(PVRef, i)
            info_str = DPCTLPlatformMgr_GetInfo(PRef, v)
            py_info = <bytes> info_str
            DPCTLCString_Delete(info_str)
            DPCTLPlatform_Delete(PRef)
            print(py_info.decode("utf-8"), end="")
    DPCTLPlatformVector_Delete(PVRef)


cpdef list get_platforms():
    """
    Returns a list of all available SYCL platforms on the system.

    Returns:
        List[:class:`.SyclPlatform`]:
            A list of SYCL platforms on the system.
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
