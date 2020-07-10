#include <dppl_oneapi_interface.hpp>
#include <CL/sycl.hpp>

using namespace dppl;
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

    rt.getCurrentContext(&ctx);
    std::cout << "==========================\n";
    std::cout << "Current context set to : \n";
    ctx->dump();

    std::cout << "==========================\n";
    std::cout << "Change the current context to : CPU 0 \n";
    rt.pushCPUContext(&ctx, 0);
    ctx->dump();
    rt.popContext();

    std::cout << "==========================\n";
    std::cout << "Try to change the current context to non-existent GPU 1 \n";
    auto ret= 0;
    if((ret = rt.pushGPUContext(&ctx, 1)) == DPPL_SUCCESS) {
        ctx->dump();
        rt.popContext();
    }
    else
        std::cout << "Sorry! no such device.\n";


    rt.setGlobalContextWithCPU(0);
    rt.getCurrentContext(&ctx);
    std::cout << "==========================\n";
    std::cout << "Current context set to : \n";
    ctx->dump();


    return 0;
}
