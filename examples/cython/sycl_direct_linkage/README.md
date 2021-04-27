# Example "sycl_direct_linkage"

This Cython extension does not use dpctl and links to SYCL directly.

It exposes `columnwise_total` function that uses oneMKL to compute
totals for each column of its argument matrix in double precision,
expected as an ordinary NumPy array in C-contiguous layout.

This functions performs the following steps:

  1. Create a SYCL queue using default device selector
  2. Creates SYCL buffer around the matrix data
  3. Creates a vector `v_ones` with all elements being ones,
     and allocates memory for the result.
  4. Calls oneMKL to compute xGEMV, as dot(v_ones, M)
  5. Returs the result as NumPy array

This extension does not allow one to control the device/queue to
which execution of kernel is being schedules.

A related example "sycl_buffer" modifies this example in that it uses
`dpctl` to retrieve the current queue, allowing a user control the queue,
and the avoid the overhead of the queue creation.

To illustrate the queue creation overhead in each call, compare execution of default queue,
which is Intel Gen9 GPU on OpenCL backend:

```
(idp) [11:24:38 ansatnuc04 sycl_direct_linkage]$ SYCL_DEVICE_FILTER=opencl:gpu python bench.py
========== Executing warm-up ==========
NumPy result:  [1. 1. 1. ... 1. 1. 1.]
SYCL(default_device) result: [1. 1. 1. ... 1. 1. 1.]
Running time of 100 calls to columnwise_total on matrix with shape (10000, 4098)
Times for default_selector, inclusive of queue creation:
[19.384219504892826, 19.49932464491576, 19.613155928440392, 19.64031868893653, 19.752969074994326]
Times for NumPy
[3.5394036192446947, 3.498957809060812, 3.4925728561356664, 3.5036555202677846, 3.493739523924887]
```

vs. timing when `dpctl`'s queue is being reused:

```
(idp) [11:29:14 ansatnuc04 sycl_buffer]$ python bench.py
========== Executing warm-up ==========
NumPy result:  [1. 1. 1. ... 1. 1. 1.]
SYCL(Intel(R) Core(TM) i7-10710U CPU @ 1.10GHz) result: [1. 1. 1. ... 1. 1. 1.]
SYCL(Intel(R) Graphics Gen9 [0x9bca]) result: [1. 1. 1. ... 1. 1. 1.]
Times for 'opencl:cpu:0'
[2.9164800881408155, 2.8714500251226127, 2.9770236839540303, 2.913622073829174, 2.7949972581118345]
Times for 'opencl:gpu:0'
[9.529508924111724, 10.288004886358976, 10.189113245811313, 10.197128206957132, 10.26169267296791]
Times for NumPy
[3.4809365631081164, 3.42917942116037, 3.42471009073779, 3.3689011191017926, 3.4336009239777923]
```

So the overhead of ``sycl::queue`` creation per call is roughly comparable with the time to
execute the actual computation.