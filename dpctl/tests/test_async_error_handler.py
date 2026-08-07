#                      Data Parallel Control (dpctl)
#
# Copyright 2026 Intel Corporation
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

"""Defines unit test cases for the default asynchronous error handler."""

import logging

import pytest

import dpctl
from dpctl._async_error_handler import _get_async_error_logger


def test_async_error_logger_is_module_logger():
    """The accessor returns the logger the handler reports through."""
    logger = _get_async_error_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "dpctl._async_error_handler"
    assert logger is logging.getLogger("dpctl._async_error_handler")


def test_queue_creation_with_async_handler_installed():
    """
    Queues are created with the async error handler installed, and remain
    usable. This guards against a NULL-handler regression crashing on wait().
    """
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Default queue could not be created")
    q.wait()
    assert q.sycl_device is not None


def test_context_creation_with_async_handler_installed():
    """
    Contexts are created with the async error handler installed. A context
    built with a NULL handler used to dereference a null function pointer if
    the SYCL runtime reported an asynchronous error against it.
    """
    try:
        ctx = dpctl.SyclContext()
    except dpctl.SyclContextCreationError:
        pytest.skip("Default context could not be created")
    assert ctx.device_count >= 1
    # Destroying the context flushes any pending asynchronous errors through
    # the handler.
    del ctx


def test_context_from_multiple_devices_async_handler():
    """
    The multi-device context constructor takes a separate code path, which
    also used to pass a NULL async error handler.
    """
    try:
        d = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default device could not be created")
    try:
        sub_devices = d.create_sub_devices(partition="max")
    except Exception:
        pytest.skip("Device could not be partitioned")
    if len(sub_devices) < 2:
        pytest.skip("Fewer than two sub-devices available")
    ctx = dpctl.SyclContext(sub_devices)
    assert ctx.device_count == len(sub_devices)


def test_async_error_handler_survives_queue_wait_on_all_devices():
    """
    Exercise queue creation and synchronization across every available
    device, so the installed handler is invoked on any backend that does
    report asynchronous errors.
    """
    devices = dpctl.get_devices()
    if not devices:
        pytest.skip("No devices available")
    for d in devices:
        try:
            q = dpctl.SyclQueue(d)
        except dpctl.SyclQueueCreationError:
            continue
        q.wait()
