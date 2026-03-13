# AGENTS.md - AI Agent Guide for DPCTL

## Purpose

This file is the top-level entry point for AI agents working in `IntelPython/dpctl`.
Use it to orient quickly, then follow directory-level `AGENTS.md` files for implementation details and local conventions.

## Repository Scope

DPCTL provides Python bindings for SYCL runtime objects and supporting infrastructure.

High-level stack:

```
Python API  ->  Cython Bindings  ->  C API (libsyclinterface)  ->  SYCL Runtime
```

## How to Work in This Repo

1. Identify the directory you are changing.
2. Read the nearest `AGENTS.md` for that directory.
3. Keep changes local and minimal; avoid unrelated refactors.
4. Validate behavior with targeted tests before broad test runs.

## Directory Guide

| Directory | Guide | Notes |
|-----------|-------|-------|
| `dpctl/` | `dpctl/AGENTS.md` | Core SYCL Python bindings and Cython patterns |
| `dpctl/memory/` | `dpctl/memory/AGENTS.md` | USM memory model and ownership rules |
| `dpctl/program/` | `dpctl/program/AGENTS.md` | Program/kernel compilation APIs |
| `dpctl/utils/` | `dpctl/utils/AGENTS.md` | Queue and utility validation helpers |
| `dpctl/tests/` | `dpctl/tests/AGENTS.md` | Test conventions and coverage expectations |
| `libsyclinterface/` | `libsyclinterface/AGENTS.md` | C API contracts and ABI-safe patterns |

## Global Constraints

- Match existing Apache 2.0 + Intel header style for source files.
- Respect style tooling from `.pre-commit-config.yaml`, `.clang-format`, and `.flake8`.
- Do not assume all devices support fp64/fp16.
- Preserve queue/device compatibility checks and explicit error paths.
- Keep memory/resource cleanup explicit and safe.

## Notes on GitHub Copilot Instructions

Files under `.github/instructions/*.instructions.md` are entry points for Copilot behavior.
They should stay concise and reference authoritative `AGENTS.md` files rather than duplicating full guidance.
