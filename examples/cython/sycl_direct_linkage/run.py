import syclbuffer as sb
import numpy as np

X = np.random.randn(20, 10)

# compute column-wise total with NumPy's own host code
print(X.sum(axis=0))

# compute column-wise total with SYCL extension
print(sb.columnwise_total(X))

