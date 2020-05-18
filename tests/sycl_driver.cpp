#include <dppy_oneapi_interface.hpp>

using namespace dppy_rt;

int main ()
{
    DppyOneAPIRuntime rt;
    rt.dump();

    std::shared_ptr<DppyOneAPIContext> ctx;
    rt.getCurrentContext(ctx);
    std::cout << "==========================\n";
    std::cout << "Current context set to : \n";
    ctx->dump();

    return 0;
}
