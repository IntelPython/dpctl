#include <dppy_oneapi_interface.hpp>
#include <CL/sycl.hpp>

using namespace dppy;
using namespace cl::sycl;

int main ()
{
    DppyOneAPIRuntime rt;
    size_t nPlats;
    std::shared_ptr<DppyOneAPIContext> ctx;

    rt.dump();

    rt.getNumPlatforms(&nPlats);
    std::cout << "==========================\n";
    std::cout << "Num platforms : " << nPlats << '\n';

    rt.getDefaultContext(&ctx);
    std::cout << "==========================\n";
    std::cout << "Current context set to : \n";
    ctx->dump();

    std::cout << "==========================\n";
    std::cout << "Change the current context to : CPU 0 \n";
    std::shared_ptr<DppyOneAPIContext> cpuCtx;
    rt.pushCPUContext(&cpuCtx, 0);
    cpuCtx->dump();

    std::cout << "==========================\n";
    std::cout << "Try to change the current context to non-existent GPU 1 \n";
    std::shared_ptr<DppyOneAPIContext> gpuCtx;
    auto ret= 0;
    if((ret = rt.pushGPUContext(&gpuCtx, 1)) == DPPY_SUCCESS)
        gpuCtx->dump();
    else
        std::cout << "Sorry! no such device\n";

    return 0;
}
