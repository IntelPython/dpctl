from __future__ import print_function

import dpctl
from dpctl import device_context, device_type


# Global runtime object inside dpctl
rt = dpctl

# Print metadata about the runtime
rt.dump()

# The runtime is initialized with a default context defined using sycl's
# default_selector. The Sycl queue for that context is returned by
# get_current_queue in a Py_Capsule.
queue = rt.get_current_queue()
print(dir(queue))

# Get a context for CPU 0 (Needs OpenCL CPU driver's). The context on exiting
# the with device_context scope gets reset to what ever context was set
# at entry of the scope. For this case, the context would go back to the
# default context
with device_context('opencl:cpu:0') as cpu_queue:
    print("========================================")
    print("Current context inside with scope")
    print("========================================")
    cpu_queue.get_sycl_device().dump_device_info()

    # Note the current context can be either directly accessed by using
    # the "cpu_queue" object, or it can be accessed via the runtime's
    # get_current_queue() function.
    print("========================================")
    print("Looking up current context using runtime")
    print("========================================")
    rt.get_current_queue().get_sycl_device().dump_device_info()


print("========================================")
print("Current context after exiting with scope")
print("========================================")
rt.get_current_queue().get_sycl_device().dump_device_info()
