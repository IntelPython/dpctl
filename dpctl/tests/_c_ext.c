//==- py_sycl-ls.c - Example of C extension working with                -===//
//  DPCTLSyclInterface C-interface library.
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2023 Intel Corporation
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
/// This file implements C Python extension using DPCTLSyclInterface library.
///
//===----------------------------------------------------------------------===//

// clang-format off
#include "Python.h"
#include "dpctl_capi.h"
// clang-format on

PyObject *py_is_sycl_queue(PyObject *self_unused, PyObject *args)
{
    PyObject *arg = NULL;
    PyObject *res = NULL;
    int status = -1;
    int check = -1;

    (void)(self_unused); // avoid unused arguments warning
    status = PyArg_ParseTuple(args, "O", &arg);
    if (!status) {
        PyErr_SetString(PyExc_TypeError, "Expecting single argument");
        return NULL;
    }

    check = PyObject_TypeCheck(arg, &PySyclQueueType);
    if (check == -1) {
        PyErr_SetString(PyExc_RuntimeError, "Type check failed");
        return NULL;
    }

    res = (check) ? Py_True : Py_False;
    Py_INCREF(res);

    return res;
}

PyObject *py_check_queue_ref(PyObject *self_unused, PyObject *args)
{
    PyObject *arg = NULL;
    PyObject *res = NULL;
    int status = -1;
    struct PySyclQueueObject *q_obj = NULL;
    DPCTLSyclQueueRef qref = NULL;

    (void)(self_unused); // avoid unused arguments warning
    status = PyArg_ParseTuple(args, "O!", &PySyclQueueType, &arg);
    if (!status) {
        PyErr_SetString(PyExc_TypeError,
                        "Expecting single argument of type dpctl.SyclQueue");
        return NULL;
    }

    q_obj = (struct PySyclQueueObject *)arg;
    qref = SyclQueue_GetQueueRef((struct PySyclQueueObject *)arg);

    res = (qref != NULL) ? Py_True : Py_False;
    Py_INCREF(res);

    return res;
}

PyObject *py_pool_roundtrip(PyObject *self_unused, PyObject *args)
{
    PyObject *arg = NULL;
    Py_ssize_t nbytes = 0;
    int status = -1;
    DPCTLSyclQueueRef qref = NULL;
    void *first = NULL;
    void *second = NULL;
    size_t first_ptr = 0;
    size_t second_ptr = 0;

    (void)(self_unused); // avoid unused arguments warning
    status = PyArg_ParseTuple(args, "O!n", &PySyclQueueType, &arg, &nbytes);
    if (!status) {
        return NULL;
    }
    qref = SyclQueue_GetQueueRef((struct PySyclQueueObject *)arg);

    if (Memory_PoolAllocate(0, DPCTL_USM_DEVICE, qref) != NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Allocated zero bytes");
        return NULL;
    }

    first = Memory_PoolAllocate((size_t)nbytes, DPCTL_USM_DEVICE, qref);
    if (first == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Pool allocation failed");
        return NULL;
    }
    first_ptr = (size_t)Memory_PoolGetPointer(first);
    Memory_PoolRelease(first);

    second = Memory_PoolAllocate((size_t)nbytes, DPCTL_USM_DEVICE, qref);
    if (second == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Pool allocation failed");
        return NULL;
    }
    second_ptr = (size_t)Memory_PoolGetPointer(second);
    Memory_PoolRelease(second);
    Memory_PoolEmptyCache();

    return Py_BuildValue("(nn)", (Py_ssize_t)first_ptr, (Py_ssize_t)second_ptr);
}

static PyMethodDef CExtMethods[] = {
    {"is_sycl_queue", py_is_sycl_queue, METH_VARARGS,
     "Checks if input object is a dpctl.SyclQueue instance"},
    {"check_queue_ref", py_check_queue_ref, METH_VARARGS,
     "Checks that queue ref obtained via C-API is not NULL"},
    {"pool_roundtrip", py_pool_roundtrip, METH_VARARGS,
     "Allocates from the memory pool twice, releasing in between, and "
     "returns both pointers"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef c_ext_module = {
    PyModuleDef_HEAD_INIT,
    "_c_ext", /* name of module */
    "",       /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CExtMethods,
    NULL,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC PyInit__c_ext(void)
{
    PyObject *m = NULL;

    import_dpctl();

    m = PyModule_Create(&c_ext_module);
    return m;
}
