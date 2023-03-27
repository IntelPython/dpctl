#pragma once
#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>

/* @brief check for overlap of memory regions behind arrays.

Presently assume that array occupies all bytes between smallest and largest
displaced elements.

TODO: Write proper Frobenius solver to account for holes, e.g.
   overlap( x_contig[::2], x_contig[1::2]) should give False,
   while this implementation gives True.
*/
namespace dpctl
{
namespace tensor
{
namespace overlap
{

struct MemoryOverlap
{
    MemoryOverlap() {}

    bool operator()(dpctl::tensor::usm_ndarray ar1,
                    dpctl::tensor::usm_ndarray ar2) const
    {
        const char *ar1_data = ar1.get_data();

        const auto &ar1_offsets = ar1.get_minmax_offsets();
        py::ssize_t ar1_elem_size =
            static_cast<py::ssize_t>(ar1.get_elemsize());

        const char *ar2_data = ar2.get_data();
        const auto &ar2_offsets = ar2.get_minmax_offsets();
        py::ssize_t ar2_elem_size =
            static_cast<py::ssize_t>(ar2.get_elemsize());

        /* Memory of array1 extends from  */
        /*    [ar1_data + ar1_offsets.first * ar1_elem_size, ar1_data +
         * ar1_offsets.second * ar1_elem_size + ar1_elem_size] */
        /* Memory of array2 extends from */
        /*    [ar2_data + ar2_offsets.first * ar2_elem_size, ar2_data +
         * ar2_offsets.second * ar2_elem_size + ar2_elem_size] */

        /* Intervals [x0, x1] and [y0, y1] do not overlap if (x0 <= x1) && (y0
         * <= y1)
         * && (x1 <=y0 || y1 <= x0 ) */
        /* Given that x0 <= x1 and y0 <= y1 are true by construction, the
         * condition for overlap us (x1 > y0) && (y1 > x0) */

        /*  Applying:
            (ar1_data + ar1_offsets.second * ar1_elem_size + ar1_elem_size >
        ar2_data
        + ar2_offsets.first * ar2_elem_size) && (ar2_data + ar2_offsets.second *
        ar2_elem_size + ar2_elem_size > ar1_data + ar1_offsets.first *
        ar1_elem_size)
        */

        auto byte_distance = static_cast<py::ssize_t>(ar2_data - ar1_data);

        py::ssize_t x1_minus_y0 =
            (-byte_distance +
             (ar1_elem_size + (ar1_offsets.second * ar1_elem_size) -
              (ar2_offsets.first * ar2_elem_size)));

        py::ssize_t y1_minus_x0 =
            (byte_distance +
             (ar2_elem_size + (ar2_offsets.second * ar2_elem_size) -
              (ar1_offsets.first * ar1_elem_size)));

        bool memory_overlap = (x1_minus_y0 > 0) && (y1_minus_x0 > 0);

        return memory_overlap;
    }
};

} // namespace overlap
} // namespace tensor
} // namespace dpctl
