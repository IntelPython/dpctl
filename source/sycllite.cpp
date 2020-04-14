#include "dp_glue.h"
#include "error_check_macros.h"
#include <CL/sycl.hpp>

int create_dp_runtime (runtime_t *rt)
{
    int err;
    runtime_t rtobj;

    rtobj = NULL;
    // Allocate a new struct dp_runtime object
    rtobj = (runtime_t)malloc(sizeof(struct dp_runtime));
    CHECK_MALLOC_ERROR(runtime_t, rt);

    rtobj->id_ = RUNTIME_ID;
    rtobj->num_platforms = 0;
    rtobj->platform_ids  = NULL;
    err = init_runtime_t_obj(rtobj);
    CHECK_DPGLUE_ERROR(err, "Could not initialize runtime object.");
    rtobj->dump_fn = dump_dp_runtime_info;

    *rt = rtobj;
#if DEBUG
    printf("DEBUG: Created an new dp_runtime object\n");
#endif
    return DP_GLUE_SUCCESS;

malloc_error:
    return DP_GLUE_FAILURE;
error:
    free(rtobj);
    return DP_GLUE_FAILURE;
}
