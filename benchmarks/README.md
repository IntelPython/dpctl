# dpctl ASV Benchmarks

Runtime-overhead benchmarks for [dpctl](https://github.com/IntelPython/dpctl)
using [ASV](https://asv.readthedocs.io/en/stable/): object construction,
queue caching, USM allocation, device enumeration, kernel bundle
compilation, data movement, kernel submission. No compute throughput.

## Coverage

| File | API |
|------|-----|
| `bench_construct.py` | `SyclDevice`/`SyclContext`/`SyclQueue`/`SyclPlatform` construction, device/queue attribute reads |
| `bench_queue_cache.py` | `get_device_cached_queue` for each key kind, vs. an uncached baseline |
| `bench_usm.py` | `MemoryUSM{Device,Host,Shared}` alloc/free, aligned alloc, first touch, USM pointer queries |
| `bench_enumerate.py` | `get_devices`, `get_num_devices`, `get_platforms`, `select_*_device`, `has_*_devices`, `select_device_with_aspects` |
| `bench_compile.py` | Kernel bundles from SPIR-V / OpenCL C source / SYCL source, kernel lookup, availability probes |
| `bench_copy.py` | `SyclQueue.memcpy`/`memcpy_async`/`fill`/`memset`, `_Memory.copy_to_host`/`copy_from_host`/`copy_from_device` |
| `bench_submit.py` | `submit`, `submit_async`, batched submission, `submit_barrier`, idle `wait` |

## Device axis

Benchmarks parameterize over the `cpu`/`gpu` filter selectors and skip when a
selector has no matching device, so the same benchmark names run on any
node. Sizes over 25% of a device's `global_mem_size` skip the same way.

## Compilation caching

`benchmarks/__init__.py` disables the persistent JIT cache
(`SYCL_CACHE_PERSISTENT=0`). `time_bundle_from_source_cold` mints a unique
kernel name per call to defeat the in-memory cache too; `_warm` reuses one
name to measure the cache-hit path.

`create_kernel_bundle_from_source` runs on the OpenCL backend only.
`create_kernel_bundle_from_sycl_source` needs a device where
`can_compile("sycl")` is true.

## Running

```bash
pip install ".[benchmark]"
cd benchmarks && asv machine --yes && asv run --python=same --quick HEAD^!
```

One module: `asv run --python=same --quick --bench bench_compile HEAD^!`

Compare commits: `asv continuous --python=same HEAD~1 HEAD`

View results: `asv publish && asv preview`
