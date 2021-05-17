#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2021 Intel Corporation
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


class Device:
    """
    Class representing Data-API concept of device.

    This is a wrapper around :class:`dpctl.SyclQueue` with custom
    formatting. The class does not have public constructor,
    but a class method to construct it from device= keyword
    in Array-API functions.

    Instance can be queried for ``sycl_queue``, ``sycl_context``,
    or ``sycl_device``.
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError("No public constructor")

    @classmethod
    def create_device(cls, dev):
        """
        Device.create_device(device)

        Creates instance of Device from argument.

        Args:
            device: None, :class:`.Device`, :class:`dpctl.SyclQueue`, or
                    a :class:`dpctl.SyclDevice` corresponding to a root
                    SYCL device.
        Raises:
            ValueError: if an instance of :class:`dpctl.SycDevice` corresponding
                        to a sub-device was specified as the argument
            SyclQueueCreationError: if :class:`dpctl.SyclQueue` could not be
                                    created from the argument
        """
        obj = super().__new__(cls)
        if isinstance(dev, Device):
            obj.sycl_queue_ = dev.sycl_queue
        elif isinstance(dev, dpctl.SyclQueue):
            obj.sycl_queue_ = dev
        elif isinstance(dev, dpctl.SyclDevice):
            par = dev.parent_device
            if par is None:
                obj.sycl_queue_ = dpctl.SyclQueue(dev)
            else:
                raise ValueError(
                    "Using non-root device {} to specify offloading "
                    "target is ambiguous. Please use dpctl.SyclQueue "
                    "targeting this device".format(dev)
                )
        else:
            if dev is None:
                obj.sycl_queue_ = dpctl.SyclQueue()
            else:
                obj.sycl_queue_ = dpctl.SyclQueue(dev)
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
        :class:`dpctl.SyclDevice` targed by this :class:`.Device`.
        """
        return self.sycl_queue_.sycl_device

    def __repr__(self):
        try:
            sd = self.sycl_device
        except AttributeError:
            raise ValueError(
                "Instance of {} is not initialized".format(self.__class__)
            )
        try:
            fs = sd.filter_string
            return "Device({})".format(fs)
        except TypeError:
            # This is a sub-device
            return repr(self.sycl_queue)
