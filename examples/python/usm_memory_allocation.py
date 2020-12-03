import dpctl
import dpctl.memory as dpmem

# allocate USM-shared byte-buffer
ms = dpmem.MemoryUSMShared(16)

# allocate USM-device byte-buffer
md = dpmem.MemoryUSMDevice(16)

# allocate USM-host byte-buffer
mh = dpmem.MemoryUSMHost(16)

# specify alignment
mda = dpmem.MemoryUSMDevice(128, alignment=16)

# allocate using given queue,
# i.e. on the device and bound to the context stored in the queue
mdq = dpmem.MemoryUSMDevice(256, queue=mda._queue)

# information about device associate with USM buffer
print("Allocation performed on device:")
mda._queue.get_sycl_device().dump_device_info()

