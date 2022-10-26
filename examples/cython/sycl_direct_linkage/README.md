# Example of sycl_direct_linkage Usage

This Cython extension does not directly use dpctl and links to SYCL.
It exposes the `columnwise_total` function that uses oneMKL to compute
totals for each column of its argument matrix in double precision
expected as an ordinary NumPy array in a C-contiguous layout.

This function performs the following steps:

  1. Creates a SYCL queue using the default device selector
  2. Creates SYCL buffer around the matrix data
  3. Creates a vector `v_ones` with all elements being ones
     and allocates memory for the result.
  4. Calls oneMKL to compute xGEMV as dot(v_ones, M)
  5. Returns the result as NumPy array

This extension does not allow to control the device or queue, to
which execution of kernel is being scheduled.

A related example "sycl_buffer" modifies this example in that it uses
`dpctl` to retrieve the current queue allowing a user to control the queue
and avoid the overhead of queue creation.

To illustrate the queue creation overhead in each call, compare the execution of the default queue,
which is Intel(R) Gen9 GPU on an OpenCL backend:

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

to the timing when the `dpctl` queue is being reused:

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

The overhead of the ``sycl::queue`` creation per call is approximately comparable with the time of
the actual computation execution.
