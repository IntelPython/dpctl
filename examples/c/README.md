# Examples C-based Python extensions using `dpctl`

The `dpctl` implements `DPCTLSyclInterface` C library as well as provides C-API to work with Python objects
and types implemented in `dpctl`. Use integration headers `dpctl_sycl_interface.h` and `dpctl_capi.h` to access
this functionality.

Use `python -m dpctl --includes` to get include compiler options and `python -m dpctl --library` to get linking options to link
to `SyclInterface` library.
