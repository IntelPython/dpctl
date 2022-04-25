from ._use_queue_device import (
    get_device_global_mem_size,
    get_device_local_mem_size,
    get_max_compute_units,
    offloaded_array_mod,
)

__all__ = [
    "get_max_compute_units",
    "get_device_global_mem_size",
    "get_device_local_mem_size",
    "offloaded_array_mod",
]
