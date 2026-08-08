---
applyTo:
  - "dpctl/memory/**"
  - "**/test_sycl_usm*.py"
---

# USM Memory Instructions

See `dpctl/memory/AGENTS.md` for details.

## USM Types
- `MemoryUSMDevice` - device-only (fastest)
- `MemoryUSMShared` - host and device accessible
- `MemoryUSMHost` - host memory, device accessible

## Lifetime Rules
1. Memory is queue-bound
2. Keep memory alive until operations complete
3. Views extend base memory lifetime
