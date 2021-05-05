#include <cstdio>

extern void main_sycl_kernels(void);
extern void main_dynamic_type_casting(void);

int main(void)
{

    main_sycl_kernels();
    std::puts(" ");
    main_dynamic_type_casting();
    return 0;
}
