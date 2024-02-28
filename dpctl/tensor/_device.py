#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2024 Intel Corporation
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
import dpctl
from dpctl._sycl_device_factory import _cached_default_device
from dpctl._sycl_queue_manager import get_device_cached_queue

__doc__ = "Implementation of array API mandated Device class"


class Device:
    """
    An object representing Data-API concept of device.

    This is a wrapper around :class:`dpctl.SyclQueue` with custom
    formatting. The class does not have public constructor,
    but a class method :meth:`dpctl.tensor.Device.create_device` to construct
    it from `device` keyword argument in Array-API functions.

    Instance can be queried for ``sycl_queue``, ``sycl_context``,
    or ``sycl_device``.
    """

    __device_queue_map__ = {}
    sycl_queue_ = None

    def __new__(cls, *args, **kwargs):
        raise TypeError("No public constructor")

    @classmethod
    def create_device(cls, device=None):
        """Device.create_device(device=None)

        Creates instance of Device from argument.

        Args:
            device:
                Device specification, i.e. `None`, :class:`.Device`,
                :class:`dpctl.SyclQueue`, or a :class:`dpctl.SyclDevice`
                corresponding to a root SYCL device.
        Raises:
            ValueError: if an instance of :class:`dpctl.SycDevice` corresponding
                        to a sub-device was specified as the argument
            SyclQueueCreationError: if :class:`dpctl.SyclQueue` could not be
                                    created from the argument
        """
        dev = device
        obj = super().__new__(cls)
        if isinstance(dev, Device):
            obj.sycl_queue_ = dev.sycl_queue
        elif isinstance(dev, dpctl.SyclQueue):
            obj.sycl_queue_ = dev
        elif isinstance(dev, dpctl.SyclDevice):
            par = dev.parent_device
            if par is None:
                obj.sycl_queue_ = get_device_cached_queue(dev)
            else:
                raise ValueError(
                    f"Using non-root device {dev} to specify offloading "
                    "target is ambiguous. Please use dpctl.SyclQueue "
                    "targeting this device"
                )
        else:
            if dev is None:
                _dev = _cached_default_device()
            else:
                _dev = dpctl.SyclDevice(dev)
            obj.sycl_queue_ = get_device_cached_queue(_dev)
        return obj

    @property
    def sycl_queue(self):
        """
        :class:`dpctl.SyclQueue` used to offload to this :class:`.Device`.
        """
        return self.sycl_queue_

    @property
    def sycl_context(self):
        """
        :class:`dpctl.SyclContext` associated with this :class:`.Device`.
        """
        return self.sycl_queue_.sycl_context

    @property
    def sycl_device(self):
        """
        :class:`dpctl.SyclDevice` targeted by this :class:`.Device`.
        """
        return self.sycl_queue_.sycl_device

    def __repr__(self):
        try:
            sd = self.sycl_device
        except AttributeError as exc:
            raise ValueError(
                f"Instance of {self.__class__} is not initialized"
            ) from exc
        try:
            fs = sd.filter_string
            return f"Device({fs})"
        except TypeError:
            # This is a sub-device
            return repr(self.sycl_queue)

    def print_device_info(self):
        "Outputs information about targeted SYCL device"
        self.sycl_device.print_device_info()

    def wait(self):
        """
        Call ``wait`` method of the underlying ``sycl_queue``.
        """
        self.sycl_queue_.wait()

    def __eq__(self, other):
        """Equality comparison based on underlying ``sycl_queue``."""
        if isinstance(other, Device):
            return self.sycl_queue.__eq__(other.sycl_queue)
        elif isinstance(other, dpctl.SyclQueue):
            return self.sycl_queue.__eq__(other)
        return False

    def __hash__(self):
        """Compute object's hash value."""
        return self.sycl_queue.__hash__()


def normalize_queue_device(sycl_queue=None, device=None):
    """normalize_queue_device(sycl_queue=None, device=None)

    Utility to process exclusive keyword arguments 'device'
    and 'sycl_queue' in functions of `dpctl.tensor`.

    Args:
        sycl_queue (:class:`dpctl.SyclQueue`, optional):
            explicitly indicates where USM allocation is done
            and the population code (if any) is executed.
            Value `None` is interpreted as get the SYCL queue
            from `device` keyword, or use default queue.
            Default: None
        device (string, :class:`dpctl.SyclDevice`, :class:`dpctl.SyclQueue,
            :class:`dpctl.tensor.Device`, optional):
            array-API keyword indicating non-partitioned SYCL device
            where array is allocated.

    Returns
        :class:`dpctl.SyclQueue` object implied by either of provided
        keywords. If both are None, `dpctl.SyclQueue()` is returned.
        If both are specified and imply the same queue, `sycl_queue`
        is returned.

    Raises:
        TypeError: if argument is not of the expected type, or keywords
            imply incompatible queues.
    """
    q = sycl_queue
    d = device
    if q is None:
        d = Device.create_device(d)
        return d.sycl_queue
    if not isinstance(q, dpctl.SyclQueue):
        raise TypeError(f"Expected dpctl.SyclQueue, got {type(q)}")
    if d is None:
        return q
    d = Device.create_device(d)
    qq = dpctl.utils.get_execution_queue(
        (
            q,
            d.sycl_queue,
        )
    )
    if qq is None:
        raise TypeError(
            "sycl_queue and device keywords can not be both specified"
        )
    return qq
