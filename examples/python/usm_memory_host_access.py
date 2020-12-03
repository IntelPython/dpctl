import dpctl
import dpctl.memory as dpmem

# USM-shared and USM-host pointers are host-accessible,
# meaning they are accessible from Python, therefore
# they implement Pyton buffer protocol

# allocate 1K of USM-shared buffer
ms = dpmem.MemoryUSMShared(1024)

# create memoryview into USM-shared buffer
msv = memoryview(ms)

# populate buffer from host one byte at a type
for i in range(len(ms)):
    ir = i % 256
    msv[i] = ir ** 2 % 256

mh = dpmem.MemoryUSMHost(64)
mhv = memoryview(mh)

# copy content of block of USM-shared buffer to
# USM-host buffer
mhv[:] = msv[78 : 78 + len(mh)]

print("Byte-values of the USM-host buffer")
print(list(mhv))

# USM-device buffer is not host accessible
md = dpmem.MemoryUSMDevice(16)
try:
    mdv = memoryview(md)
except Exception as e:
    print("")
    print(
        "An expected exception was raised during attempted construction of memoryview from USM-device memory object."
    )
    print("\t", e)
