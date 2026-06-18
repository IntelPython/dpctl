//==- py_sycl-ls.c - Example of C extension working with                -===//
//  DPCTLSyclInterface C-interface library.
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2022 Intel Corporation
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
#include "dpctl_sycl_interface.h"
// clang-format on

PyObject *sycl_ls(PyObject *self_unused, PyObject *args)
{
    DPCTLPlatformVectorRef PVRef = NULL;
    size_t psz = 0;

    (void)(self_unused); // avoid unused arguments warning
    (void)(args);
    PVRef = DPCTLPlatform_GetPlatforms();

    if (PVRef) {
        psz = DPCTLPlatformVector_Size(PVRef);

        for (size_t i = 0; i < psz; ++i) {
            DPCTLSyclPlatformRef PRef = DPCTLPlatformVector_GetAt(PVRef, i);
            const char *pl_info = DPCTLPlatformMgr_GetInfo(PRef, 2);

            printf("Platform: %ld::\n%s\n", i, pl_info);

            DPCTLCString_Delete(pl_info);
            DPCTLPlatform_Delete(PRef);
        }

        DPCTLPlatformVector_Delete(PVRef);
    }

    Py_RETURN_NONE;
}

static PyMethodDef SyclLSMethods[] = {
    {"sycl_ls", sycl_ls, METH_NOARGS, "Output information about SYCL platform"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static int syclls_module_exec(PyObject *m)
{
    (void)(m);
    import_dpctl();
    if (PyErr_Occurred()) {
        return -1;
    }
    return 0;
}

static PyModuleDef_Slot syclls_module_slots[] = {
    {Py_mod_exec, syclls_module_exec},
#if PY_VERSION_HEX >= 0x030D0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}};

static struct PyModuleDef syclls_module = {
    PyModuleDef_HEAD_INIT,
    "_py_sycl_ls", /* name of module */
    "",            /* module documentation, may be NULL */
    0,             /* size of per-interpreter state of the module */
    SyclLSMethods,
    syclls_module_slots,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC PyInit__py_sycl_ls(void)
{
    return PyModuleDef_Init(&syclls_module);
}
