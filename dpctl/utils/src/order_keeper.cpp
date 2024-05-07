#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sequential_order_keeper.hpp"
#include <sycl/sycl.hpp>

PYBIND11_MODULE(_seq_order_keeper, m)
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
}
