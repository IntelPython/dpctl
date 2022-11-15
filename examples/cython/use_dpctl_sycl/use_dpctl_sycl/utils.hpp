#pragma once

#include <CL/sycl.hpp>
#include <string>

std::string get_device_name(sycl::device d)
{
    return d.get_info<sycl::info::device::name>();
}

std::string get_device_driver_version(sycl::device d)
{
    return d.get_info<sycl::info::device::driver_version>();
}

sycl::device *copy_device(const sycl::device &d)
{
    auto copy_ptr = new sycl::device(d);
    return copy_ptr;
}
