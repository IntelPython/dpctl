.. _contributor_guides_syclinterface_memory_ownership:

Working with DPCTLSyclInterface library
=======================================

The DPCLSyclInterface library is a C-API library which does provide select C++ functions
for casting from C opaque pointers to pointers to corresponding C++ classes.

This document explains the memory ownership model adopted by DPCTLSyclInterface.

Function declarations are decorated with tokens such as ``__dpctl_keep``, ``__dpctl_take``,
and ``__dpctl_give``. Use of these tokens in declarations serves to self-document memory
ownership semantics.

Token ``__dpctl_give`` indicates that the function makes a new allocation and delegates
responsibility to free it to the caller. Creation functions, such as
:c:func:`DCPTLDevice_Create`, belong to category of such functions.

The token ``__dpctl_take`` indicates that the library deletes the allocation associated
with the object to which the token applies. Deletion functions, such as
:c:func:`DPCTLDevice_Delete`, represent set of such functions.

The token ``__dpctl_keep`` indicates that the library does not alter allocation associated
with the object to which the tocken applies. Functions to query integral device descriptors,
such as :c:func:`DPCTLDevice_GetMaxComputeUnits`, are examples of such functions.

.. code-block:: C
    :caption: Example: Example of use of DPCTLSyclInterface functions

    // filename: example_syclinterface.c
    #include "stdint.h"
    #include "stdio.h"
    #include "dpctl_sycl_interface.h"

    int main(void) {
        // we own memory allocation associated DRef object
        DPCTLSyclDeviceRef DRef = DPCTLDevice_Create();

        // we own memory allocation associated with char array
        const char* name = DPCTLDevice_GetName(DRef);
        uint32_t cu = DPCTLDevice_GetMaxComputeUnits(DRef);

        // Free allocations associated with DRef
        DPCTLDevice_Delete(DRef);

        printf("Device %s has %d compute units\n", name, cu);

        // Free memory allocate for device name
        DPCTLCString_Delete(name);

        return 0;
    }

Building the example:

.. code-block:: bash
    :caption: Building the example into an executable

    icx example_syclinterface.c -fsanitize=address                 \
        $(python -m dpctl --includes) $(python -m dpctl --library) \
        -o example

Running the example displays the following output without errors:

.. code-block:: text
    :caption: Execution of the executable and its output

    $ ./a.x
    Device Intel(R) Graphics [0x9a49] has 96 compute units
