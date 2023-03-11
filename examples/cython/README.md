# Examples of data-parallel Python extensions written in Cython

The `dpctl` package provides Cython definition files for types it defines.

Use `cimport dpctl as c_dpctl`, `cimport dpctl.memory as c_dpm`, or `cimport dpctl.tensor as c_dpt`
to use these definitions.

Cython definition fille `dpctl.sycl` provides incomplete definitions of core SYCL runtime classes as
well as conversion routine between `SyclInterface` reference types and SYCL runtime classes.
