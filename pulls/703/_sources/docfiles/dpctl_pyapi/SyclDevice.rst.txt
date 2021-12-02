.. _SyclDevice_api:

################
dpctl.SyclDevice
################

.. currentmodule:: dpctl

.. autoclass:: SyclDevice

    .. rubric:: Attributes:

    .. autoautosummary:: dpctl.SyclDevice
        :attributes:

    .. rubric:: Public methods:

    .. autoautosummary:: dpctl.SyclDevice
        :methods:

Detail
======

Attributes
----------

.. autoattribute:: dpctl.SyclDevice.backend
.. autoattribute:: dpctl.SyclDevice.default_selector_score
.. autoattribute:: dpctl.SyclDevice.device_type
.. autoattribute:: dpctl.SyclDevice.driver_version
.. autoattribute:: dpctl.SyclDevice.filter_string
.. autoattribute:: dpctl.SyclDevice.has_aspect_accelerator
.. autoattribute:: dpctl.SyclDevice.has_aspect_cpu
.. autoattribute:: dpctl.SyclDevice.has_aspect_custom
.. autoattribute:: dpctl.SyclDevice.has_aspect_fp16
.. autoattribute:: dpctl.SyclDevice.has_aspect_fp64
.. autoattribute:: dpctl.SyclDevice.has_aspect_gpu
.. autoattribute:: dpctl.SyclDevice.has_aspect_host
.. autoattribute:: dpctl.SyclDevice.has_aspect_image
.. autoattribute:: dpctl.SyclDevice.has_aspect_int64_base_atomics
.. autoattribute:: dpctl.SyclDevice.has_aspect_int64_extended_atomics
.. autoattribute:: dpctl.SyclDevice.has_aspect_online_compiler
.. autoattribute:: dpctl.SyclDevice.has_aspect_online_linker
.. autoattribute:: dpctl.SyclDevice.has_aspect_queue_profiling
.. autoattribute:: dpctl.SyclDevice.has_aspect_usm_device_allocations
.. autoattribute:: dpctl.SyclDevice.has_aspect_usm_host_allocations
.. autoattribute:: dpctl.SyclDevice.has_aspect_usm_restricted_shared_allocations
.. autoattribute:: dpctl.SyclDevice.has_aspect_usm_shared_allocations
.. autoattribute:: dpctl.SyclDevice.has_aspect_usm_system_allocator
.. autoattribute:: dpctl.SyclDevice.image_2d_max_height
.. autoattribute:: dpctl.SyclDevice.image_2d_max_width
.. autoattribute:: dpctl.SyclDevice.image_3d_max_depth
.. autoattribute:: dpctl.SyclDevice.image_3d_max_height
.. autoattribute:: dpctl.SyclDevice.image_3d_max_width
.. autoattribute:: dpctl.SyclDevice.is_accelerator
.. autoattribute:: dpctl.SyclDevice.is_cpu
.. autoattribute:: dpctl.SyclDevice.is_gpu
.. autoattribute:: dpctl.SyclDevice.is_host
.. autoattribute:: dpctl.SyclDevice.max_compute_units
.. autoattribute:: dpctl.SyclDevice.max_num_sub_groups
.. autoattribute:: dpctl.SyclDevice.max_read_image_args
.. autoattribute:: dpctl.SyclDevice.max_work_group_size
.. autoattribute:: dpctl.SyclDevice.max_work_item_dims
.. autoattribute:: dpctl.SyclDevice.max_work_item_sizes
.. autoattribute:: dpctl.SyclDevice.max_write_image_args
.. autoattribute:: dpctl.SyclDevice.name
.. autoattribute:: dpctl.SyclDevice.parent_device
.. autoattribute:: dpctl.SyclDevice.preferred_vector_width_char
.. autoattribute:: dpctl.SyclDevice.preferred_vector_width_double
.. autoattribute:: dpctl.SyclDevice.preferred_vector_width_float
.. autoattribute:: dpctl.SyclDevice.preferred_vector_width_half
.. autoattribute:: dpctl.SyclDevice.preferred_vector_width_int
.. autoattribute:: dpctl.SyclDevice.preferred_vector_width_long
.. autoattribute:: dpctl.SyclDevice.preferred_vector_width_short
.. autoattribute:: dpctl.SyclDevice.sub_group_independent_forward_progress
.. autoattribute:: dpctl.SyclDevice.vendor

Public methods
--------------

.. autofunction:: dpctl.SyclDevice.addressof_ref
.. autofunction:: dpctl.SyclDevice.create_sub_devices
.. autofunction:: dpctl.SyclDevice.get_filter_string
.. autofunction:: dpctl.SyclDevice.print_device_info
