---
applyTo:
  - "libsyclinterface/**/*.h"
  - "libsyclinterface/**/*.hpp"
  - "libsyclinterface/**/*.cpp"
---

# C API Instructions

See [libsyclinterface/AGENTS.md](/libsyclinterface/AGENTS.md) for conventions.

## Quick Reference

### Naming
`DPCTL<ClassName>_<MethodName>` (e.g., `DPCTLDevice_Create`)

### Ownership annotations (from `Support/MemOwnershipAttrs.h`)
- `__dpctl_give` - caller must free
- `__dpctl_take` - function takes ownership
- `__dpctl_keep` - function only observes

### Key Rules
- Annotate all parameters and returns
- Return NULL on failure
- Use `DPCTL_API` for exports
