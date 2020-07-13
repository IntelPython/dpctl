from __future__ import print_function
from dppl import runtime, device_context, device_type

# Global runtime object inside dppl
rt = runtime

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
with device_context(device_type.cpu, 0) as cpu_queue:
    print("========================================")
    print("Current context inside with scope")
    print("========================================")
    rt.dump_queue(cpu_queue)
    
    # Note the current context can be either directly accessed by using
    # the "cpu_queue" object, or it can be accessed via the runtime's
    # get_current_queue() function.
    print("========================================")
    print("Looking up current context using runtime")
    print("========================================")
    rt.dump_queue(rt.get_current_queue())


print("========================================")
print("Current context after exiting with scope")
print("========================================")
rt.dump_queue(rt.get_current_queue())
