
#include "../helper/include/dpctl_vector_macros.h"
#include "Support/MemOwnershipAttrs.h"

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(vector_class<SYCLREF(EL)>, VECTOR(EL))
}
__dpctl_give VECTOR(EL) FN(EL, Create)()
{
    try {
        auto Vec = new vector_class<SYCLREF(EL)>();
        return wrap(Vec);
    } catch (std::bad_alloc const &ba) {
        return nullptr;
    }
}

void FN(EL, Delete)(__dpctl_take VECTOR(EL) VRef)
{
    delete unwrap(VRef);
}

void FN(EL, Clear)(__dpctl_keep VECTOR(EL) VRef)
{
    auto Vec = unwrap(VRef);

    for (auto i = 0ul; i < Vec->size(); ++i) {
        auto D = unwrap((*Vec)[i]);
        delete D;
    }
    Vec->clear();
}

size_t FN(EL, Size)(__dpctl_keep VECTOR(EL) VRef)
{
    return unwrap(VRef)->size();
}

SYCLREF(EL) FN(EL, GetAt)(__dpctl_keep VECTOR(EL) VRef, size_t index)
{
    auto Vec = unwrap(VRef);
    SYCLREF(EL) ret = nullptr;
    try {
        ret = Vec->at(index);
    } catch (std::out_of_range const &oor) {
        std::cerr << oor.what() << '\n';
    }
    return ret;
}
