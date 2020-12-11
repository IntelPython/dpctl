# Example "sycl_direct_linkage"

This Cython extension does not use dpCtl and links to SYCL directly.

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
`dpCtl` to retrieve the current queue, allowing a user control the queue,
and the avoid the overhead of the queue creation