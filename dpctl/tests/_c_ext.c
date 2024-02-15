//==- py_sycl-ls.c - Example of C extension working with                -===//
//  DPCTLSyclInterface C-interface library.
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2022-2024 Intel Corporation
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

PyObject *py_is_usm_ndarray(PyObject *self_unused, PyObject *args)
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

    check = PyObject_TypeCheck(arg, &PyUSMArrayType);
    if (check == -1) {
        PyErr_SetString(PyExc_RuntimeError, "Type check failed");
        return NULL;
    }

    res = (check) ? Py_True : Py_False;
    Py_INCREF(res);

    return res;
}

PyObject *py_usm_ndarray_ndim(PyObject *self_unused, PyObject *args)
{
    PyObject *arg = NULL;
    struct PyUSMArrayObject *array = NULL;
    PyObject *res = NULL;
    int status = -1;
    int ndim = -1;

    (void)(self_unused); // avoid unused arguments warning
    status = PyArg_ParseTuple(args, "O!", &PyUSMArrayType, &arg);
    if (!status) {
        PyErr_SetString(
            PyExc_TypeError,
            "Expecting single argument of type dpctl.tensor.usm_ndarray");
        return NULL;
    }

    array = (struct PyUSMArrayObject *)arg;
    ndim = UsmNDArray_GetNDim(array);

    res = PyLong_FromLong(ndim);
    return res;
}

static PyMethodDef CExtMethods[] = {
    {"is_usm_ndarray", py_is_usm_ndarray, METH_VARARGS,
     "Checks if input object is an usm_ndarray instance"},
    {"usm_ndarray_ndim", py_usm_ndarray_ndim, METH_VARARGS,
     "Get ndim property of an usm_ndarray instance"},
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
