# libsyclinterface/ - C API Layer

## Purpose

C wrapper around SYCL C++ runtime for language interoperability (used by Cython bindings).

## Directory Structure

```
libsyclinterface/
├── include/syclinterface/
│   ├── Support/
│   │   ├── DllExport.h         # DPCTL_API macro
│   │   ├── ExternC.h           # extern "C" macros
│   │   └── MemOwnershipAttrs.h # Ownership annotations
│   └── dpctl_sycl_*.h          # API headers
├── source/                      # Implementation
└── tests/                       # C interface tests
```

## Naming Convention

```
DPCTL<ClassName>_<MethodName>
```

Examples: `DPCTLDevice_Create`, `DPCTLQueue_Submit`, `DPCTLContext_Delete`

## Memory Ownership Annotations

Defined in `Support/MemOwnershipAttrs.h`:

| Annotation | Meaning |
|------------|---------|
| `__dpctl_give` | Caller receives ownership, must free |
| `__dpctl_take` | Function takes ownership, caller must not use after |
| `__dpctl_keep` | Function only observes, does not take ownership |
| `__dpctl_null` | NULL is valid value |

## Function Pattern

```c
DPCTL_API
__dpctl_give DPCTLSyclContextRef
DPCTLContext_Create(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                    error_handler_callback *handler,
                    int properties);

DPCTL_API
void DPCTLContext_Delete(__dpctl_take DPCTLSyclContextRef CRef);
```

## Key Rules

- Always annotate ownership on parameters and returns
- Return NULL on failure
- Use `DPCTL_C_EXTERN_C_BEGIN/END` for C++ implementations
- Use `DPCTL_API` for exported functions
