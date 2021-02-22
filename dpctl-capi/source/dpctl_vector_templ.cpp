
#include "../helper/include/dpctl_vector_macros.h"
#include "Support/MemOwnershipAttrs.h"

__dpctl_give VECTOR(EL) FN(EL, _Create)()
{
    try {
        auto Vec = new vector_class<VECTOR(EL)>();
        return wrap(Vec);
    } catch (std::bad_alloc const &ba) {
        return nullptr;
    }
}

void FN(EL, _Delete)(__dpctl_take VECTOR(EL) VRef)
{
    delete unwrap(VRef);
}

void FN(EL, _Clear)(__dpctl_keep VECTOR(EL) VRef)
{
    auto Vec = unwrap(VRef);

    for (auto i = 0ul; i < Vec->size(); ++i) {
        auto D = unwrap((*Vec)[i]);
        delete D;
    }
    Vec->clear();
}

size_t FN(EL, _Size)(__dpctl_keep VECTOR(EL) VRef)
{
    return unwrap(VRef)->size();
}

size_t FN(EL, _GetAt)(__dpctl_keep VECTOR(EL) VRef, size_t index)
{
    auto Vec = unwrap(VRef);
    EL ret = nullptr;
    try {
        ret = Vec->at(index);
    } catch (std::out_of_range const &oor) {
        std::cerr << oor.what() << '\n';
    }
    return ret;
}
