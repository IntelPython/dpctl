import dpctl
import dpctl.memory as dpmem
import numpy as np

ms = dpmem.MemoryUSMShared(32)
md = dpmem.MemoryUSMDevice(32)

host_buf = np.random.randint(0, 42, dtype=np.uint8, size=32)

# copy host byte-like object to USM-device buffer
md.copy_from_host(host_buf)

# copy USM-device buffer to USM-shared buffer in parallel (using sycl::queue::memcpy)
ms.copy_from_device(md)

# build numpy array reusing host-accessible USM-shared memory
X = np.ndarray((len(ms),), buffer=ms, dtype=np.uint8)

# Display Python object NumPy ndarray is viewing into
print("numpy.ndarray.base: ", X.base)
print("")

# Print content of the view
print("View..........: ", X)

# Print content of the original host buffer
print("host_buf......: ", host_buf)

# use copy_to_host to retrieve memory of USM-device memory
print("copy_to_host(): ", md.copy_to_host())
