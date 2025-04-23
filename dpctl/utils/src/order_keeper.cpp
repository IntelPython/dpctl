#include "dpctl4pybind11.hpp"
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sequential_order_keeper.hpp"
#include <sycl/sycl.hpp>

PYBIND11_MODULE(_seq_order_keeper, m, py::mod_gil_not_used())
{
    py::class_<SequentialOrder>(m, "_OrderManager")
        .def(py::init<std::size_t>())
        .def(py::init<>())
        .def(py::init<SequentialOrder>())
        .def("get_num_submitted_events",
             &SequentialOrder::get_num_submitted_events)
        .def("get_num_host_task_events",
             &SequentialOrder::get_num_host_task_events)
        .def("get_submitted_events", &SequentialOrder::get_submitted_events)
        .def("get_host_task_events", &SequentialOrder::get_host_task_events)
        .def("add_to_both_events", &SequentialOrder::add_to_both_events)
        .def("add_vector_to_both_events",
             &SequentialOrder::add_vector_to_both_events)
        .def("add_to_host_task_events",
             &SequentialOrder::add_to_host_task_events)
        .def("add_to_submitted_events",
             &SequentialOrder::add_to_submitted_events)
        .def("wait", &SequentialOrder::wait,
             py::call_guard<py::gil_scoped_release>());

    auto submit_empty_task_fn =
        [](sycl::queue &exec_q,
           const std::vector<sycl::event> &depends) -> sycl::event {
        return exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);
            cgh.single_task([]() {
                // empty body
            });
        });
    };
    m.def("_submit_empty_task", submit_empty_task_fn, py::arg("sycl_queue"),
          py::arg("depends") = py::list());
}
