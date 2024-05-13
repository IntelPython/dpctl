//=== sorting.hpp -  Implementation of sorting kernels       ---*-C++-*--/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for tensor sort/argsort operations.
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <functional>
#include <iterator>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/sorting/sort_detail.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{

namespace sort_detail
{

/*! @brief Merge two contiguous sorted segments */
template <typename InAcc, typename OutAcc, typename Compare>
void merge_impl(const std::size_t offset,
                const InAcc in_acc,
                OutAcc out_acc,
                const std::size_t start_1,
                const std::size_t end_1,
                const std::size_t end_2,
                const std::size_t start_out,
                Compare comp,
                const std::size_t chunk)
{
    const std::size_t start_2 = end_1;
    // Borders of the sequences to merge within this call
    const std::size_t local_start_1 = sycl::min(offset + start_1, end_1);
    const std::size_t local_end_1 = sycl::min(local_start_1 + chunk, end_1);
    const std::size_t local_start_2 = sycl::min(offset + start_2, end_2);
    const std::size_t local_end_2 = sycl::min(local_start_2 + chunk, end_2);

    const std::size_t local_size_1 = local_end_1 - local_start_1;
    const std::size_t local_size_2 = local_end_2 - local_start_2;

    const auto r_item_1 = in_acc[end_1 - 1];
    const auto l_item_2 = (start_2 < end_2) ? in_acc[start_2] : r_item_1;

    // Copy if the sequences are sorted with respect to each other or merge
    // otherwise
    if (!comp(l_item_2, r_item_1)) {
        const std::size_t out_shift_1 = start_out + local_start_1 - start_1;
        const std::size_t out_shift_2 =
            start_out + end_1 - start_1 + local_start_2 - start_2;

        for (std::size_t i = 0; i < local_size_1; ++i) {
            out_acc[out_shift_1 + i] = in_acc[local_start_1 + i];
        }
        for (std::size_t i = 0; i < local_size_2; ++i) {
            out_acc[out_shift_2 + i] = in_acc[local_start_2 + i];
        }
    }
    else if (comp(r_item_1, l_item_2)) {
        const std::size_t out_shift_1 =
            start_out + end_2 - start_2 + local_start_1 - start_1;
        const std::size_t out_shift_2 = start_out + local_start_2 - start_2;
        for (std::size_t i = 0; i < local_size_1; ++i) {
            out_acc[out_shift_1 + i] = in_acc[local_start_1 + i];
        }
        for (std::size_t i = 0; i < local_size_2; ++i) {
            out_acc[out_shift_2 + i] = in_acc[local_start_2 + i];
        }
    }
    // Perform merging
    else {

        // Process 1st sequence
        if (local_start_1 < local_end_1) {
            // Reduce the range for searching within the 2nd sequence and handle
            // bound items find left border in 2nd sequence
            const auto local_l_item_1 = in_acc[local_start_1];
            std::size_t l_search_bound_2 =
                lower_bound_impl(in_acc, start_2, end_2, local_l_item_1, comp);
            const std::size_t l_shift_1 = local_start_1 - start_1;
            const std::size_t l_shift_2 = l_search_bound_2 - start_2;

            out_acc[start_out + l_shift_1 + l_shift_2] = local_l_item_1;

            std::size_t r_search_bound_2{};
            // find right border in 2nd sequence
            if (local_size_1 > 1) {
                const auto local_r_item_1 = in_acc[local_end_1 - 1];
                r_search_bound_2 = lower_bound_impl(
                    in_acc, l_search_bound_2, end_2, local_r_item_1, comp);
                const auto r_shift_1 = local_end_1 - 1 - start_1;
                const auto r_shift_2 = r_search_bound_2 - start_2;

                out_acc[start_out + r_shift_1 + r_shift_2] = local_r_item_1;
            }

            // Handle intermediate items
            if (r_search_bound_2 == l_search_bound_2) {
                const std::size_t shift_2 = l_search_bound_2 - start_2;
                for (std::size_t idx = local_start_1 + 1; idx < local_end_1 - 1;
                     ++idx) {
                    const auto intermediate_item_1 = in_acc[idx];
                    const std::size_t shift_1 = idx - start_1;
                    out_acc[start_out + shift_1 + shift_2] =
                        intermediate_item_1;
                }
            }
            else {
                for (std::size_t idx = local_start_1 + 1; idx < local_end_1 - 1;
                     ++idx) {
                    const auto intermediate_item_1 = in_acc[idx];
                    // we shouldn't seek in whole 2nd sequence. Just for the
                    // part where the 1st sequence should be
                    l_search_bound_2 = lower_bound_impl(
                        in_acc, l_search_bound_2, r_search_bound_2,
                        intermediate_item_1, comp);
                    const std::size_t shift_1 = idx - start_1;
                    const std::size_t shift_2 = l_search_bound_2 - start_2;

                    out_acc[start_out + shift_1 + shift_2] =
                        intermediate_item_1;
                }
            }
        }
        // Process 2nd sequence
        if (local_start_2 < local_end_2) {
            // Reduce the range for searching within the 1st sequence and handle
            // bound items find left border in 1st sequence
            const auto local_l_item_2 = in_acc[local_start_2];
            std::size_t l_search_bound_1 =
                upper_bound_impl(in_acc, start_1, end_1, local_l_item_2, comp);
            const std::size_t l_shift_1 = l_search_bound_1 - start_1;
            const std::size_t l_shift_2 = local_start_2 - start_2;

            out_acc[start_out + l_shift_1 + l_shift_2] = local_l_item_2;

            std::size_t r_search_bound_1{};
            // find right border in 1st sequence
            if (local_size_2 > 1) {
                const auto local_r_item_2 = in_acc[local_end_2 - 1];
                r_search_bound_1 = upper_bound_impl(
                    in_acc, l_search_bound_1, end_1, local_r_item_2, comp);
                const std::size_t r_shift_1 = r_search_bound_1 - start_1;
                const std::size_t r_shift_2 = local_end_2 - 1 - start_2;

                out_acc[start_out + r_shift_1 + r_shift_2] = local_r_item_2;
            }

            // Handle intermediate items
            if (l_search_bound_1 == r_search_bound_1) {
                const std::size_t shift_1 = l_search_bound_1 - start_1;
                for (auto idx = local_start_2 + 1; idx < local_end_2 - 1; ++idx)
                {
                    const auto intermediate_item_2 = in_acc[idx];
                    const std::size_t shift_2 = idx - start_2;
                    out_acc[start_out + shift_1 + shift_2] =
                        intermediate_item_2;
                }
            }
            else {
                for (auto idx = local_start_2 + 1; idx < local_end_2 - 1; ++idx)
                {
                    const auto intermediate_item_2 = in_acc[idx];
                    // we shouldn't seek in whole 1st sequence. Just for the
                    // part where the 2nd sequence should be
                    l_search_bound_1 = upper_bound_impl(
                        in_acc, l_search_bound_1, r_search_bound_1,
                        intermediate_item_2, comp);
                    const std::size_t shift_1 = l_search_bound_1 - start_1;
                    const std::size_t shift_2 = idx - start_2;

                    out_acc[start_out + shift_1 + shift_2] =
                        intermediate_item_2;
                }
            }
        }
    }
}

namespace
{
template <typename Iter, typename Compare>
void insertion_sort_impl(Iter first,
                         const size_t begin,
                         const size_t end,
                         Compare comp)
{
    for (size_t i = begin + 1; i < end; ++i) {
        const auto val_i = first[i];
        size_t j = i - 1;
        while ((j + 1 > begin) && (comp(val_i, first[j]))) {
            first[j + 1] = first[j];
            --j;
        }
        if (j + 1 < i) {
            first[j + 1] = val_i;
        }
    }
}

template <typename Iter, typename Compare>
void bubble_sort_impl(Iter first,
                      const size_t begin,
                      const size_t end,
                      Compare comp)
{
    if (begin < end) {
        for (size_t i = begin; i < end; ++i) {
            // Handle intermediate items
            for (size_t idx = i + 1; idx < end; ++idx) {
                if (comp(first[idx], first[i])) {
                    std::swap(first[i], first[idx]);
                }
            }
        }
    }
}

template <typename Iter, typename Compare>
void leaf_sort_impl(Iter first,
                    const size_t begin,
                    const size_t end,
                    Compare comp)
{
    return insertion_sort_impl<Iter, Compare>(
        std::move(first), std::move(begin), std::move(end), std::move(comp));
}
} // namespace

template <typename Iter> struct GetValueType
{
    using value_type = typename std::iterator_traits<Iter>::value_type;
};

template <typename ElementType,
          sycl::access::address_space Space,
          sycl::access::decorated IsDecorated>
struct GetValueType<sycl::multi_ptr<ElementType, Space, IsDecorated>>
{
    using value_type = ElementType;
};

template <typename ElementType,
          int Dim,
          sycl::access_mode Mode,
          sycl::target Target,
          sycl::access::placeholder isPlaceholder>
struct GetValueType<
    sycl::accessor<ElementType, Dim, Mode, Target, isPlaceholder>>
{
    using value_type = ElementType;
};

template <typename ElementType, int Dim, typename AllocatorT>
struct GetValueType<sycl::buffer<ElementType, Dim, AllocatorT>>
{
    using value_type = ElementType;
};

template <typename Iter> struct GetReadOnlyAccess
{
    Iter operator()(Iter it, sycl::handler &)
    {
        return it;
    }
};

template <typename ElementType, int Dim, typename AllocatorT>
struct GetReadOnlyAccess<sycl::buffer<ElementType, Dim, AllocatorT>>
{
    auto operator()(sycl::buffer<ElementType, Dim, AllocatorT> buf,
                    sycl::handler &cgh)
    {
        sycl::accessor acc(buf, cgh, sycl::read_only);
        return acc;
    }
};

template <typename Iter> struct GetWriteDiscardAccess
{
    Iter operator()(Iter it, sycl::handler &)
    {
        return it;
    }
};

template <typename ElementType, int Dim, typename AllocatorT>
struct GetWriteDiscardAccess<sycl::buffer<ElementType, Dim, AllocatorT>>
{
    auto operator()(sycl::buffer<ElementType, Dim, AllocatorT> buf,
                    sycl::handler &cgh)
    {
        sycl::accessor acc(buf, cgh, sycl::write_only, sycl::no_init);
        return acc;
    }
};

template <typename Iter> struct GetReadWriteAccess
{
    Iter operator()(Iter it, sycl::handler &)
    {
        return it;
    }
};

template <typename ElementType, int Dim, typename AllocatorT>
struct GetReadWriteAccess<sycl::buffer<ElementType, Dim, AllocatorT>>
{
    auto operator()(sycl::buffer<ElementType, Dim, AllocatorT> buf,
                    sycl::handler &cgh)
    {
        sycl::accessor acc(buf, cgh, sycl::read_write);
        return acc;
    }
};

template <typename T1, typename T2, typename Comp>
class sort_base_step_contig_krn;

template <typename InpAcc, typename OutAcc, typename Comp>
sycl::event
sort_base_step_contig_impl(sycl::queue &q,
                           const size_t iter_nelems,
                           const size_t sort_nelems,
                           const InpAcc input,
                           OutAcc output,
                           const Comp &comp,
                           const size_t conseq_nelems_sorted,
                           const std::vector<sycl::event> &depends = {})
{

    using inpT = typename GetValueType<InpAcc>::value_type;
    using outT = typename GetValueType<OutAcc>::value_type;
    using KernelName = sort_base_step_contig_krn<inpT, outT, Comp>;

    const size_t n_segments =
        quotient_ceil<size_t>(sort_nelems, conseq_nelems_sorted);

    sycl::event base_sort = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const sycl::range<1> gRange{iter_nelems * n_segments};

        auto input_acc = GetReadOnlyAccess<InpAcc>{}(input, cgh);
        auto output_acc = GetWriteDiscardAccess<OutAcc>{}(output, cgh);

        cgh.parallel_for<KernelName>(gRange, [=](sycl::id<1> id) {
            const size_t iter_id = id[0] / n_segments;
            const size_t segment_id = id[0] - iter_id * n_segments;

            const size_t iter_offset = iter_id * sort_nelems;
            const size_t beg_id =
                iter_offset + segment_id * conseq_nelems_sorted;
            const size_t end_id =
                iter_offset +
                std::min<size_t>((segment_id + 1) * conseq_nelems_sorted,
                                 sort_nelems);
            for (size_t i = beg_id; i < end_id; ++i) {
                output_acc[i] = input_acc[i];
            }

            leaf_sort_impl(output_acc, beg_id, end_id, comp);
        });
    });

    return base_sort;
}

template <typename T1, typename T2, typename Comp>
class sort_over_work_group_contig_krn;

template <typename InpAcc, typename OutAcc, typename Comp>
sycl::event
sort_over_work_group_contig_impl(sycl::queue &q,
                                 size_t iter_nelems,
                                 size_t sort_nelems,
                                 const InpAcc input,
                                 OutAcc output,
                                 const Comp &comp,
                                 size_t &nelems_wg_sorts,
                                 const std::vector<sycl::event> &depends = {})
{
    using inpT = typename GetValueType<InpAcc>::value_type;
    using T = typename GetValueType<OutAcc>::value_type;
    using KernelName = sort_over_work_group_contig_krn<inpT, T, Comp>;

    const auto &kernel_id = sycl::get_kernel_id<KernelName>();

    auto const &ctx = q.get_context();
    auto const &dev = q.get_device();
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        ctx, {dev}, {kernel_id});

    auto krn = kb.template get_kernel(kernel_id);

    const std::uint32_t max_sg_size = krn.template get_info<
        sycl::info::kernel_device_specific::max_sub_group_size>(dev);
    const std::uint64_t device_local_memory_size =
        dev.get_info<sycl::info::device::local_mem_size>();

    //  leave 512 bytes of local memory for RT
    const std::uint64_t safety_margin = 512;

    const std::uint64_t nelems_per_slm =
        (device_local_memory_size - safety_margin) / (2 * sizeof(T));

    constexpr std::uint32_t sub_groups_per_work_group = 4;
    const std::uint32_t elems_per_wi = dev.has(sycl::aspect::cpu) ? 8 : 2;

    const size_t lws = sub_groups_per_work_group * max_sg_size;

    nelems_wg_sorts = elems_per_wi * lws;

    if (nelems_wg_sorts > nelems_per_slm) {
        nelems_wg_sorts = (q.get_device().has(sycl::aspect::cpu) ? 16 : 4);

        return sort_base_step_contig_impl<InpAcc, OutAcc, Comp>(
            q, iter_nelems, sort_nelems, input, output, comp, nelems_wg_sorts,
            depends);
    }

    // This assumption permits doing away with using a loop
    assert(nelems_wg_sorts % lws == 0);

    const size_t n_segments =
        quotient_ceil<size_t>(sort_nelems, nelems_wg_sorts);

    sycl::event base_sort_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.use_kernel_bundle(kb);

        sycl::range<1> global_range{iter_nelems * n_segments * lws};
        sycl::range<1> local_range{lws};

        sycl::range<1> slm_range{nelems_wg_sorts};
        sycl::local_accessor<T, 1> work_space(slm_range, cgh);
        sycl::local_accessor<T, 1> scratch_space(slm_range, cgh);

        auto input_acc = GetReadOnlyAccess<InpAcc>{}(input, cgh);
        auto output_acc = GetWriteDiscardAccess<OutAcc>{}(output, cgh);

        sycl::nd_range<1> ndRange(global_range, local_range);

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> it) {
            const size_t group_id = it.get_group_linear_id();
            const size_t iter_id = group_id / n_segments;
            const size_t segment_id = group_id - iter_id * n_segments;
            const size_t lid = it.get_local_linear_id();

            const size_t segment_start_idx = segment_id * nelems_wg_sorts;
            const size_t segment_end_idx = std::min<size_t>(
                segment_start_idx + nelems_wg_sorts, sort_nelems);
            const size_t wg_chunk_size = segment_end_idx - segment_start_idx;

            // load input into SLM
            for (size_t array_id = segment_start_idx + lid;
                 array_id < segment_end_idx; array_id += lws)
            {
                T v = (array_id < sort_nelems)
                          ? input_acc[iter_id * sort_nelems + array_id]
                          : T{};
                work_space[array_id - segment_start_idx] = v;
            }
            sycl::group_barrier(it.get_group());

            const size_t chunk = quotient_ceil<size_t>(nelems_wg_sorts, lws);

            const size_t chunk_start_idx = lid * chunk;
            const size_t chunk_end_idx =
                sycl::min(chunk_start_idx + chunk, wg_chunk_size);

            leaf_sort_impl(work_space, chunk_start_idx, chunk_end_idx, comp);

            sycl::group_barrier(it.get_group());

            bool data_in_temp = false;
            size_t n_chunks_merged = 1;

            // merge chunk while n_chunks_merged * chunk < wg_chunk_size
            const size_t max_chunks_merged = 1 + ((wg_chunk_size - 1) / chunk);
            for (; n_chunks_merged < max_chunks_merged;
                 data_in_temp = !data_in_temp, n_chunks_merged *= 2)
            {
                const size_t nelems_sorted_so_far = n_chunks_merged * chunk;
                const size_t q = (lid / n_chunks_merged);
                const size_t start_1 =
                    sycl::min(2 * nelems_sorted_so_far * q, wg_chunk_size);
                const size_t end_1 =
                    sycl::min(start_1 + nelems_sorted_so_far, wg_chunk_size);
                const size_t end_2 =
                    sycl::min(end_1 + nelems_sorted_so_far, wg_chunk_size);
                const size_t offset = chunk * (lid - q * n_chunks_merged);

                if (data_in_temp) {
                    merge_impl(offset, scratch_space, work_space, start_1,
                               end_1, end_2, start_1, comp, chunk);
                }
                else {
                    merge_impl(offset, work_space, scratch_space, start_1,
                               end_1, end_2, start_1, comp, chunk);
                }
                sycl::group_barrier(it.get_group());
            }

            const auto &out_src = (data_in_temp) ? scratch_space : work_space;
            for (size_t array_id = segment_start_idx + lid;
                 array_id < segment_end_idx; array_id += lws)
            {
                if (array_id < sort_nelems) {
                    output_acc[iter_id * sort_nelems + array_id] =
                        out_src[array_id - segment_start_idx];
                }
            }
        });
    });

    return base_sort_ev;
}

class vacuous_krn;

inline sycl::event tie_events(sycl::queue &q,
                              const std::vector<sycl::event> depends)
{
    if (depends.empty())
        return sycl::event();
    if (depends.size() == 1)
        return depends[0];

    sycl::event e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        using KernelName = vacuous_krn;
        cgh.single_task<KernelName>([]() {});
    });

    return e;
}

template <typename T, typename Comp> class merge_adjacent_blocks_to_temp_krn;

template <typename T, typename Comp> class merge_adjacent_blocks_from_temp_krn;

template <typename Acc, typename Comp>
sycl::event
merge_sorted_block_contig_impl(sycl::queue &q,
                               size_t iter_nelems,
                               size_t sort_nelems,
                               Acc output,
                               const Comp comp,
                               size_t sorted_block_size,
                               const std::vector<sycl::event> &depends = {})
{

    if (sorted_block_size >= sort_nelems)
        return tie_events(q, depends);

    // experimentally determined value
    // size of segments worked upon by each work-item during merging
    const sycl::device &dev = q.get_device();
    const size_t segment_size = (dev.has(sycl::aspect::cpu)) ? 32 : 4;

    const size_t chunk_size =
        (sorted_block_size < segment_size) ? sorted_block_size : segment_size;

    assert(sorted_block_size % chunk_size == 0);

    using T = typename GetValueType<Acc>::value_type;

    sycl::buffer<T, 1> temp_buf(sycl::range<1>{iter_nelems * sort_nelems});
    // T *allocated_mem = sycl::malloc_device<T>(iter_nelems * sort_nelems, q);

    bool needs_copy = true;
    bool used_depends = false;

    sycl::event dep_ev;
    size_t chunks_merged = sorted_block_size / chunk_size;

    assert(!(chunks_merged & (chunks_merged - 1)));

    using ToTempKernelName = class merge_adjacent_blocks_to_temp_krn<T, Comp>;
    using FromTempKernelName =
        class merge_adjacent_blocks_from_temp_krn<T, Comp>;

    while (chunks_merged * chunk_size < sort_nelems) {
        sycl::event local_dep = dep_ev;

        sycl::event merge_ev = q.submit([&](sycl::handler &cgh) {
            if (used_depends) {
                cgh.depends_on(local_dep);
            }
            else {
                cgh.depends_on(depends);
                used_depends = true;
            }

            const size_t n_chunks =
                quotient_ceil<size_t>(sort_nelems, chunk_size);

            if (needs_copy) {
                sycl::accessor temp_acc{temp_buf, cgh, sycl::write_only,
                                        sycl::no_init};
                auto output_acc = GetReadOnlyAccess<Acc>{}(output, cgh);
                cgh.parallel_for<ToTempKernelName>(
                    {iter_nelems * n_chunks}, [=](sycl::id<1> wid) {
                        auto flat_idx = wid[0];
                        auto iter_idx = flat_idx / n_chunks;
                        auto idx = flat_idx - n_chunks * iter_idx;

                        const std::size_t idx_mult =
                            (idx / chunks_merged) * chunks_merged;
                        const std::size_t idx_rem = (idx - idx_mult);
                        const std::size_t start_1 =
                            sycl::min(2 * idx_mult * chunk_size, sort_nelems);
                        const std::size_t end_1 = sycl::min(
                            start_1 + chunks_merged * chunk_size, sort_nelems);
                        const std::size_t end_2 = sycl::min(
                            end_1 + chunks_merged * chunk_size, sort_nelems);
                        const std::size_t offset = chunk_size * idx_rem;

                        const std::size_t iter_offset = iter_idx * sort_nelems;

                        merge_impl(offset, output_acc, temp_acc,
                                   iter_offset + start_1, iter_offset + end_1,
                                   iter_offset + end_2, iter_offset + start_1,
                                   comp, chunk_size);
                    });
            }
            else {
                sycl::accessor temp_acc{temp_buf, cgh, sycl::read_only};
                auto output_acc = GetWriteDiscardAccess<Acc>{}(output, cgh);
                cgh.parallel_for<FromTempKernelName>(
                    {iter_nelems * n_chunks}, [=](sycl::id<1> wid) {
                        auto flat_idx = wid[0];
                        auto iter_idx = flat_idx / n_chunks;
                        auto idx = flat_idx - n_chunks * iter_idx;

                        const std::size_t idx_mult =
                            (idx / chunks_merged) * chunks_merged;
                        const std::size_t idx_rem = (idx - idx_mult);
                        const std::size_t start_1 =
                            sycl::min(2 * idx_mult * chunk_size, sort_nelems);
                        const std::size_t end_1 = sycl::min(
                            start_1 + chunks_merged * chunk_size, sort_nelems);
                        const std::size_t end_2 = sycl::min(
                            end_1 + chunks_merged * chunk_size, sort_nelems);
                        const std::size_t offset = chunk_size * idx_rem;

                        const std::size_t iter_offset = iter_idx * sort_nelems;

                        merge_impl(offset, temp_acc, output_acc,
                                   iter_offset + start_1, iter_offset + end_1,
                                   iter_offset + end_2, iter_offset + start_1,
                                   comp, chunk_size);
                    });
            }
        });

        chunks_merged *= 2;
        dep_ev = merge_ev;

        if (chunks_merged * chunk_size < sort_nelems) {
            needs_copy = !needs_copy;
        }
    }

    if (needs_copy) {
        sycl::event copy_ev = q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dep_ev);

            sycl::accessor temp_acc{temp_buf, cgh, sycl::read_only};
            auto output_acc = GetWriteDiscardAccess<Acc>{}(output, cgh);

            cgh.copy(temp_acc, output_acc);
        });
        dep_ev = copy_ev;
    }

    return dep_ev;
}

} // end of namespace sort_detail

typedef sycl::event (*sort_contig_fn_ptr_t)(sycl::queue &,
                                            size_t,
                                            size_t,
                                            const char *,
                                            char *,
                                            ssize_t,
                                            ssize_t,
                                            ssize_t,
                                            ssize_t,
                                            const std::vector<sycl::event> &);

template <typename argTy, typename Comp = std::less<argTy>>
sycl::event stable_sort_axis1_contig_impl(
    sycl::queue &exec_q,
    size_t iter_nelems, // number of sub-arrays to sort (num. of rows in a
                        // matrix when sorting over rows)
    size_t sort_nelems, // size of each array to sort  (length of rows, i.e.
                        // number of columns)
    const char *arg_cp,
    char *res_cp,
    ssize_t iter_arg_offset,
    ssize_t iter_res_offset,
    ssize_t sort_arg_offset,
    ssize_t sort_res_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + sort_arg_offset;
    argTy *res_tp =
        reinterpret_cast<argTy *>(res_cp) + iter_res_offset + sort_res_offset;

    auto comp = Comp{};

    constexpr size_t sequential_sorting_threshold = 64;

    if (sort_nelems < sequential_sorting_threshold) {
        // equal work-item sorts entire row
        sycl::event sequential_sorting_ev =
            sort_detail::sort_base_step_contig_impl<const argTy *, argTy *,
                                                    Comp>(
                exec_q, iter_nelems, sort_nelems, arg_tp, res_tp, comp,
                sort_nelems, depends);

        return sequential_sorting_ev;
    }
    else {
        size_t sorted_block_size{};

        // Sort segments of the array
        sycl::event base_sort_ev =
            sort_detail::sort_over_work_group_contig_impl<const argTy *,
                                                          argTy *, Comp>(
                exec_q, iter_nelems, sort_nelems, arg_tp, res_tp, comp,
                sorted_block_size, // modified in place with size of sorted
                                   // block size
                depends);

        // Merge segments in parallel until all elements are sorted
        sycl::event merges_ev =
            sort_detail::merge_sorted_block_contig_impl<argTy *, Comp>(
                exec_q, iter_nelems, sort_nelems, res_tp, comp,
                sorted_block_size, {base_sort_ev});

        return merges_ev;
    }
}

template <typename T1, typename T2, typename T3>
class populate_indexed_data_krn;

template <typename T1, typename T2, typename T3> class index_write_out_krn;

template <typename pairT, typename ValueComp> struct TupleComp
{
    bool operator()(const pairT &p1, const pairT &p2) const
    {
        const ValueComp value_comp{};
        return value_comp(std::get<0>(p1), std::get<0>(p2));
    }
};

template <typename argTy,
          typename IndexTy,
          typename ValueComp = std::less<argTy>>
sycl::event stable_argsort_axis1_contig_impl(
    sycl::queue &exec_q,
    size_t iter_nelems, // number of sub-arrays to sort (num. of rows in a
                        // matrix when sorting over rows)
    size_t sort_nelems, // size of each array to sort  (length of rows, i.e.
                        // number of columns)
    const char *arg_cp,
    char *res_cp,
    ssize_t iter_arg_offset,
    ssize_t iter_res_offset,
    ssize_t sort_arg_offset,
    ssize_t sort_res_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + sort_arg_offset;
    IndexTy *res_tp =
        reinterpret_cast<IndexTy *>(res_cp) + iter_res_offset + sort_res_offset;

    using ValueIndexT = std::pair<argTy, IndexTy>;
    const TupleComp<ValueIndexT, ValueComp> tuple_comp{};

    static constexpr size_t determine_automatically = 0;
    size_t sorted_block_size =
        (sort_nelems >= 512) ? 512 : determine_automatically;

    sycl::buffer<ValueIndexT, 1> indexed_data(
        sycl::range<1>(iter_nelems * sort_nelems));
    sycl::buffer<ValueIndexT, 1> temp_buf(
        sycl::range<1>(iter_nelems * sort_nelems));

    sycl::event populate_indexed_data_ev =
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);
            sycl::accessor acc(indexed_data, cgh, sycl::write_only,
                               sycl::no_init);

            auto const &range = indexed_data.get_range();

            using KernelName =
                populate_indexed_data_krn<argTy, IndexTy, ValueComp>;

            cgh.parallel_for<KernelName>(range, [=](sycl::id<1> id) {
                size_t i = id[0];
                size_t sort_id = i % sort_nelems;
                acc[i] =
                    std::make_pair(arg_tp[i], static_cast<IndexTy>(sort_id));
            });
        });

    // Sort segments of the array
    sycl::event base_sort_ev = sort_detail::sort_over_work_group_contig_impl(
        exec_q, iter_nelems, sort_nelems, indexed_data, temp_buf, tuple_comp,
        sorted_block_size, // modified in place with size of sorted block size
        {populate_indexed_data_ev});

    // Merge segments in parallel until all elements are sorted
    sycl::event merges_ev = sort_detail::merge_sorted_block_contig_impl(
        exec_q, iter_nelems, sort_nelems, temp_buf, tuple_comp,
        sorted_block_size, {base_sort_ev});

    sycl::event write_out_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(merges_ev);

        auto temp_acc =
            sort_detail::GetReadOnlyAccess<decltype(temp_buf)>{}(temp_buf, cgh);

        using KernelName = index_write_out_krn<argTy, IndexTy, ValueComp>;

        cgh.parallel_for<KernelName>(temp_buf.get_range(), [=](sycl::id<1> id) {
            res_tp[id] = std::get<1>(temp_acc[id]);
        });
    });

    return write_out_ev;
}

} // end of namespace kernels
} // end of namespace tensor
} // end of namespace dpctl
