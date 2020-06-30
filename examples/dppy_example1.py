from __future__ import print_function
from dppy import runtime, device_context, device_type

# Global runtime object inside dppy
rt = runtime

# Print metadata about the runtime
rt.dump()

# The runtime is initialized with a default context defined using sycl's
# default_selector
ctx = rt.get_current_context()

# Get a PyCapsule for the actual sycl::queue encapsulated by the ctx object.
# Note that the PyCapsule contains a shared_ptr<cl::sycl::queue> * that the
# caller now owns.
queue = ctx.get_sycl_queue()
print(dir(queue))


# Print metadata about the runtime
print("========================================")
print("Current context")
print("========================================")
ctx.dump()


# Get a context for CPU 0 (Needs OpenCL CPU driver's). The context on exiting
# the with device_context scope gets reset to what ever context was set
# at entry of the scope. For this case, the context would go back to the
# default context
with device_context(device_type.cpu, 0) as cpuctx:
    print("========================================")
    print("Current context inside with scope")
    print("========================================")
    cpuctx.dump()
    
    # Note the current context can be either directly accessed by using 
    # the "cpuctx" object, or it can be accessed via the runtime's 
    # get_current_context() function.
    print("========================================")
    print("Looking up current context using runtime")
    print("========================================")
    rt.get_current_context().dump()


print("========================================")
print("Current context after exiting with scope")
print("========================================")
rt.get_current_context().dump()