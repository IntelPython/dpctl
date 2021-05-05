#pragma once

#include "dpctl_sycl_types.h"
#include <cstdlib>

namespace usm_array
{

class strided_array
{
public:
    strided_array() {}
    explicit strided_array(char *ptr, int nd, size_t *shape, int typenum)
        : ptr_(ptr), nd_(nd), shape_(shape), typenum_(typenum){};
    explicit strided_array(char *ptr,
                           int nd,
                           size_t *shape,
                           std::ptrdiff_t *strides,
                           int typenum)
        : ptr_(ptr), nd_(nd), shape_(shape), strides_(strides),
          typenum_(typenum){};
    explicit strided_array(char *ptr,
                           int nd,
                           size_t *shape,
                           std::ptrdiff_t *strides,
                           int typenum,
                           int flags)
        : ptr_(ptr), nd_(nd), shape_(shape), strides_(strides),
          typenum_(typenum), flags_(flags){};
    strided_array(const strided_array &other) = default;
    strided_array(strided_array &&other) = default;
    ~strided_array() = default;

    // member access functions
    char *get_data_ptr() const
    {
        return ptr_;
    }
    int ndim() const
    {
        return nd_;
    }
    size_t *get_shape_ptr() const
    {
        return shape_;
    }
    std::ptrdiff_t *get_strides_ptr() const
    {
        return strides_;
    }
    int typenum() const
    {
        return typenum_;
    }
    int flags() const
    {
        return flags_;
    }

    size_t get_shape(int i) const
    {
        return shape_[i];
    }
    std::ptrdiff_t get_stride(int i) const
    {
        return strides_[i];
    }

private:
    char *ptr_{0};
    int nd_{0};
    size_t *shape_{0};
    std::ptrdiff_t *strides_{0};
    int typenum_{0};
    int flags_{0};
};

class usm_array : public strided_array
{
public:
    explicit usm_array(char *data,
                       int nd,
                       size_t *shape,
                       std::ptrdiff_t *strides,
                       int typenum,
                       int flags,
                       DPCTLSyclQueueRef qref)
        : strided_array(data, nd, shape, strides, typenum, flags), q_(qref){};

    usm_array(const usm_array &other) = default;
    usm_array(usm_array &&other) = default;
    ~usm_array() = default;

    DPCTLSyclQueueRef get_queue_ref() const
    {
        return q_;
    }

private:
    DPCTLSyclQueueRef q_{0};
};

} // namespace usm_array
