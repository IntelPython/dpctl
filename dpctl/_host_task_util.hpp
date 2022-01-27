#include "Python.h"
#include "syclinterface/dpctl_data_types.h"
#include <CL/sycl.hpp>

int async_dec_ref(DPCTLSyclQueueRef QRef,
                  PyObject **obj_array,
                  size_t obj_array_size,
                  DPCTLSyclEventRef *ERefs,
                  size_t nERefs)
{

    sycl::queue *q = reinterpret_cast<sycl::queue *>(QRef);

    std::vector<PyObject *> obj_vec;
    obj_vec.reserve(obj_array_size);
    for (size_t obj_id = 0; obj_id < obj_array_size; ++obj_id) {
        obj_vec.push_back(obj_array[obj_id]);
    }

    try {
        q->submit([&](sycl::handler &cgh) {
            for (size_t ev_id = 0; ev_id < nERefs; ++ev_id) {
                cgh.depends_on(
                    *(reinterpret_cast<sycl::event *>(ERefs[ev_id])));
            }
            cgh.host_task([obj_array_size, obj_vec]() {
                {
                    PyGILState_STATE gstate;
                    gstate = PyGILState_Ensure();
                    for (size_t i = 0; i < obj_array_size; ++i) {
                        Py_DECREF(obj_vec[i]);
                    }
                    PyGILState_Release(gstate);
                }
            });
        });
    } catch (const std::exception &e) {
        return 1;
    }

    return 0;
}
