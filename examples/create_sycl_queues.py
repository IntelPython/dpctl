from __future__ import print_function

import dppl

# Print metadata about the runtime
dppl.dump()

# The runtime is initialized with a default context defined using sycl's
# default_selector. The Sycl queue for that context is returned by
# get_current_queue in a Py_Capsule.
queue = dppl.get_current_queue()

print("========================================")
print("Current context before entering with scope")
print("========================================")
dppl.dump_queue_info(queue)

# Get a context for CPU 0 (Needs OpenCL CPU driver's). The context on exiting
# the with device_context scope gets reset to what ever context was set
# at entry of the scope. For this case, the context would go back to the
# default context
with dppl.device_context(dppl.device_type.cpu, 0) as cpu_queue:
    print("========================================")
    print("Current context inside with scope")
    print("========================================")
    dppl.dump_queue_info(cpu_queue)
    
    # Note the current context can be either directly accessed by using
    # the "cpu_queue" object, or it can be accessed via the runtime's
    # get_current_queue() function.
    print("========================================")
    print("Looking up current context using runtime")
    print("========================================")
    dppl.dump_queue_info(dppl.get_current_queue())


print("========================================")
print("Current context after exiting with scope")
print("========================================")
dppl.dump_queue_info(dppl.get_current_queue())
