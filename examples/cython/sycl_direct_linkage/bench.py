import dpctl
import syclbuffer_naive as sb
import numpy as np

X = np.full((10 ** 4, 4098), 1e-4, dtype="d")

# warm-up
print("=" * 10 + " Executing warm-up " + "=" * 10)
print("NumPy result: ", X.sum(axis=0))

print(
    "SYCL(default_device) result: {}".format(
        sb.columnwise_total(X),
    )
)

import timeit

print("Running time of 100 calls to columnwise_total on matrix with shape {}".format(X.shape))

print("Times for default_selector, inclusive of queue creation:")
print(
    timeit.repeat(
        stmt="sb.columnwise_total(X)",
        setup="sb.columnwise_total(X)",  # ensure JIT compilation is not counted
        number=100,
        globals=globals(),
    )
)

print("Times for NumPy")
print(timeit.repeat(stmt="X.sum(axis=0)", number=100, globals=globals()))
