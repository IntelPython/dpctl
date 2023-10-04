#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace
{

std::uint32_t py_intel_device_id(const sycl::device &d)
{
    static constexpr std::uint32_t device_id_unavailable = 0;

    if (d.has(sycl::aspect::ext_intel_device_id)) {
        return d.get_info<sycl::ext::intel::info::device::device_id>();
    }

    return device_id_unavailable;
}

std::uint32_t py_intel_gpu_eu_count(const sycl::device &d)
{
    static constexpr std::uint32_t eu_count_unavailable = 0;

    if (d.has(sycl::aspect::ext_intel_gpu_eu_count)) {
        return d.get_info<sycl::ext::intel::info::device::gpu_eu_count>();
    }

    return eu_count_unavailable;
}

std::uint32_t py_intel_gpu_hw_threads_per_eu(const sycl::device &d)
{
    static constexpr std::uint32_t thread_count_unavailable = 0;

    if (d.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
        return d
            .get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
    }

    return thread_count_unavailable;
}

std::uint32_t py_intel_gpu_eu_simd_width(const sycl::device &d)
{
    static constexpr std::uint32_t width_unavailable = 0;

    if (d.has(sycl::aspect::ext_intel_gpu_eu_simd_width)) {
        return d.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
    }

    return width_unavailable;
}

std::uint32_t py_intel_gpu_slices(const sycl::device &d)
{
    static constexpr std::uint32_t count_unavailable = 0;

    if (d.has(sycl::aspect::ext_intel_gpu_slices)) {
        return d.get_info<sycl::ext::intel::info::device::gpu_slices>();
    }

    return count_unavailable;
}

std::uint32_t py_intel_gpu_subslices_per_slice(const sycl::device &d)
{
    static constexpr std::uint32_t count_unavailable = 0;

    if (d.has(sycl::aspect::ext_intel_gpu_subslices_per_slice)) {
        return d.get_info<
            sycl::ext::intel::info::device::gpu_subslices_per_slice>();
    }

    return count_unavailable;
}

std::uint32_t py_intel_gpu_eu_count_per_subslice(const sycl::device &d)
{
    static constexpr std::uint32_t count_unavailable = 0;

    if (d.has(sycl::aspect::ext_intel_gpu_eu_count_per_subslice)) {
        return d.get_info<
            sycl::ext::intel::info::device::gpu_eu_count_per_subslice>();
    }

    return count_unavailable;
}

std::uint64_t py_intel_max_mem_bandwidth(const sycl::device &d)
{
    static constexpr std::uint64_t bandwidth_unavailable = 0;

    if (d.has(sycl::aspect::ext_intel_max_mem_bandwidth)) {
        return d.get_info<sycl::ext::intel::info::device::max_mem_bandwidth>();
    }

    return bandwidth_unavailable;
}

}; // namespace

PYBIND11_MODULE(_device_queries, m)
{
    m.def("intel_device_info_device_id", &py_intel_device_id,
          "Get ext_intel_device_id for the device, zero if not an intel device",
          py::arg("device"));

    m.def("intel_device_info_gpu_eu_count", &py_intel_gpu_eu_count,
          "Returns the number of execution units (EUs) associated with the "
          "Intel GPU.",
          py::arg("device"));

    m.def("intel_device_info_gpu_hw_threads_per_eu",
          &py_intel_gpu_hw_threads_per_eu,
          "Returns the number of hardware threads in EU.", py::arg("device"));

    m.def("intel_device_info_gpu_eu_simd_width", &py_intel_gpu_eu_simd_width,
          "Returns the physical SIMD width of the execution unit (EU).",
          py::arg("device"));

    m.def("intel_device_info_gpu_slices", &py_intel_gpu_slices,
          "Returns the number of slices in the GPU device, or zero.",
          py::arg("device"));

    m.def("intel_device_info_gpu_subslices_per_slice",
          &py_intel_gpu_subslices_per_slice,
          "Returns the number of subslices per slice.", py::arg("device"));

    m.def("intel_device_info_gpu_eu_count_per_subslice",
          &py_intel_gpu_eu_count_per_subslice,
          "Returns the number of EUs per subslice of GPU.", py::arg("device"));

    m.def("intel_device_info_max_mem_bandwidth", &py_intel_max_mem_bandwidth,
          "Returns the maximum memory bandwidth in units of bytes/second.",
          py::arg("device"));
}
