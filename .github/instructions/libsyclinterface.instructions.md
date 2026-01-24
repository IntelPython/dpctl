---
applyTo:
  - "libsyclinterface/**/*.h"
  - "libsyclinterface/**/*.hpp"
  - "libsyclinterface/**/*.cpp"
  - "libsyclinterface/**/*.c"
---

# C API Layer Instructions (libsyclinterface)

## Context

The libsyclinterface directory provides a C API wrapper around SYCL C++ runtime objects. This enables language interoperability with Python (via Cython) and other languages.

## Directory Structure

```
libsyclinterface/
├── include/syclinterface/
│   ├── Support/
│   │   ├── DllExport.h         # DLL export macros
│   │   ├── ExternC.h           # extern "C" wrapper macros
│   │   └── MemOwnershipAttrs.h # Memory ownership annotations
│   ├── dpctl_sycl_context_interface.h
│   ├── dpctl_sycl_device_interface.h
│   ├── dpctl_sycl_device_manager.h
│   ├── dpctl_sycl_device_selector_interface.h
│   ├── dpctl_sycl_event_interface.h
│   ├── dpctl_sycl_kernel_bundle_interface.h
│   ├── dpctl_sycl_kernel_interface.h
│   ├── dpctl_sycl_platform_interface.h
│   ├── dpctl_sycl_platform_manager.h
│   ├── dpctl_sycl_queue_interface.h
│   ├── dpctl_sycl_queue_manager.h
│   ├── dpctl_sycl_usm_interface.h
│   └── dpctl_sycl_types.h
├── helper/
│   ├── include/
│   └── source/
├── source/                      # Implementation files
└── tests/                       # C interface tests
```

## Function Naming Convention

```
DPCTL<ClassName>_<MethodName>
```

### Examples
```c
DPCTLDevice_Create()
DPCTLDevice_Delete()
DPCTLDevice_GetName()
DPCTLDevice_GetVendor()
DPCTLDevice_HasAspect()

DPCTLQueue_Create()
DPCTLQueue_Delete()
DPCTLQueue_Submit()
DPCTLQueue_Memcpy()
DPCTLQueue_Wait()

DPCTLContext_Create()
DPCTLContext_Delete()
DPCTLContext_GetDeviceCount()
```

## Memory Ownership Annotations

Defined in `Support/MemOwnershipAttrs.h`:

| Annotation | Meaning | Caller Action |
|------------|---------|---------------|
| `__dpctl_give` | Function returns ownership | Caller must eventually free |
| `__dpctl_take` | Function takes ownership | Caller must not use after call |
| `__dpctl_keep` | Function only observes | Caller retains ownership |
| `__dpctl_null` | NULL is valid | Check for NULL returns |

### Usage Examples

```c
// Caller receives ownership - must call Delete later
DPCTL_API
__dpctl_give DPCTLSyclContextRef
DPCTLContext_Create(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                    error_handler_callback *handler,
                    int properties);

// Function takes ownership - don't use CRef after this
DPCTL_API
void DPCTLContext_Delete(__dpctl_take DPCTLSyclContextRef CRef);

// Function only observes - QRef still valid after call
DPCTL_API
size_t DPCTLQueue_GetBackend(__dpctl_keep const DPCTLSyclQueueRef QRef);

// May return NULL on failure
DPCTL_API
__dpctl_give __dpctl_null DPCTLSyclDeviceRef
DPCTLDevice_Create(__dpctl_keep DPCTLSyclDeviceSelectorRef DSRef);
```

## Opaque Pointer Types

```c
// Forward declarations - actual struct defined in implementation
typedef struct DPCTLOpaqueSyclContext *DPCTLSyclContextRef;
typedef struct DPCTLOpaqueSyclDevice *DPCTLSyclDeviceRef;
typedef struct DPCTLOpaqueSyclEvent *DPCTLSyclEventRef;
typedef struct DPCTLOpaqueSyclKernel *DPCTLSyclKernelRef;
typedef struct DPCTLOpaqueSyclPlatform *DPCTLSyclPlatformRef;
typedef struct DPCTLOpaqueSyclQueue *DPCTLSyclQueueRef;
```

## Extern C Wrapper

Use macros from `Support/ExternC.h`:

```c
#include "Support/ExternC.h"

DPCTL_C_EXTERN_C_BEGIN

// All C API declarations here
DPCTL_API
__dpctl_give DPCTLSyclDeviceRef
DPCTLDevice_Create(...);

DPCTL_C_EXTERN_C_END
```

## DLL Export

Use `DPCTL_API` macro from `Support/DllExport.h`:

```c
DPCTL_API
__dpctl_give DPCTLSyclContextRef
DPCTLContext_Create(...);
```

## Error Handling

### Error Callback

```c
// Type definition
typedef void error_handler_callback(int err_code);

// Usage in async error handling
DPCTL_API
__dpctl_give DPCTLSyclQueueRef
DPCTLQueue_Create(__dpctl_keep DPCTLSyclContextRef CRef,
                  __dpctl_keep DPCTLSyclDeviceRef DRef,
                  error_handler_callback *handler,
                  int properties);
```

### NULL Return Convention

Functions that create objects return `NULL` on failure:

```c
DPCTLSyclDeviceRef dref = DPCTLDevice_Create(selector);
if (dref == NULL) {
    // Handle error
}
```

## Header File Template

```c
//===-- dpctl_sycl_foo_interface.h - C API for Foo        -*-C-*-===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// ...
//===----------------------------------------------------------------------===//
///
/// \file
/// This header declares a C API for sycl::foo class.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/**
 * @defgroup FooInterface Foo class C wrapper
 */

/*!
 * @brief Creates a new Foo object.
 *
 * @param    param1    Description of param1.
 * @param    param2    Description of param2.
 * @return   A new DPCTLSyclFooRef on success, NULL on failure.
 * @ingroup  FooInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclFooRef
DPCTLFoo_Create(__dpctl_keep DPCTLSyclBarRef BRef);

/*!
 * @brief Deletes a Foo object.
 *
 * @param    FRef    The Foo reference to delete.
 * @ingroup  FooInterface
 */
DPCTL_API
void DPCTLFoo_Delete(__dpctl_take DPCTLSyclFooRef FRef);

DPCTL_C_EXTERN_C_END
```

## Implementation Pattern

```cpp
// In source/dpctl_sycl_foo_interface.cpp

#include "dpctl_sycl_foo_interface.h"
#include <sycl/sycl.hpp>

namespace
{
// Internal helpers in anonymous namespace
struct DPCTLOpaqueSyclFoo
{
    sycl::foo *ptr;
};

inline sycl::foo *unwrap(DPCTLSyclFooRef ref)
{
    return ref ? ref->ptr : nullptr;
}

inline DPCTLSyclFooRef wrap(sycl::foo *ptr)
{
    if (ptr) {
        auto ref = new DPCTLOpaqueSyclFoo;
        ref->ptr = ptr;
        return ref;
    }
    return nullptr;
}
} // namespace

__dpctl_give DPCTLSyclFooRef
DPCTLFoo_Create(__dpctl_keep DPCTLSyclBarRef BRef)
{
    try {
        auto bar = unwrap(BRef);
        if (!bar) return nullptr;

        auto foo = new sycl::foo(*bar);
        return wrap(foo);
    }
    catch (const std::exception &e) {
        // Log error
        return nullptr;
    }
}

void DPCTLFoo_Delete(__dpctl_take DPCTLSyclFooRef FRef)
{
    if (FRef) {
        delete FRef->ptr;
        delete FRef;
    }
}
```

## Best Practices

1. **Always use ownership annotations** - Every function parameter and return value should have appropriate annotation
2. **Check for NULL** - Always validate input pointers before use
3. **Use try/catch** - Catch C++ exceptions and return NULL or error codes
4. **Wrap/unwrap consistently** - Use helper functions for opaque pointer conversion
5. **Document with Doxygen** - Every public function needs documentation
6. **Thread safety** - Document thread safety guarantees
