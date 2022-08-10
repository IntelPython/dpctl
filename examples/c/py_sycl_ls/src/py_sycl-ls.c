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
#include "syclinterface/dpctl_sycl_platform_interface.h"
#include "syclinterface/dpctl_sycl_platform_manager.h"
#include "syclinterface/dpctl_utils.h"
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

static struct PyModuleDef syclls_module = {
    PyModuleDef_HEAD_INIT,
    "_py_sycl_ls", /* name of module */
    "",            /* module documentation, may be NULL */
    -1,            /* size of per-interpreter state of the module,
                      or -1 if the module keeps state in global variables. */
    SyclLSMethods,
    NULL,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC PyInit__py_sycl_ls(void)
{
    PyObject *m;

    import_dpctl();

    m = PyModule_Create(&syclls_module);

    return m;
}
