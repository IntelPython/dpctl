import syclbuffer as sb
import numpy as np

X = np.random.randn(100, 4)

print("Result computed by NumPy")
print(X.sum(axis=0))
print("Result computed by SYCL extension")
print(sb.columnwise_total(X))


print("")
# controlling where to offload
import dpctl

with dpctl.device_context('opencl:gpu'):
    print("Running on: ", dpctl.get_current_queue().get_sycl_device().get_device_name())
    print(sb.columnwise_total(X))

with dpctl.device_context('opencl:cpu'):
    print("Running on: ", dpctl.get_current_queue().get_sycl_device().get_device_name())
    print(sb.columnwise_total(X))
