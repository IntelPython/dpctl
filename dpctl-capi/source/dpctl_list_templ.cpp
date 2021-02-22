

__dpctl_give VECTOR(EL) FN(EL, _Create)() {}

void FN(EL, _Delete)(__dpctl_take VECTOR(EL) VRef) {}

void FN(EL, _Clear)(__dpctl_keep VECTOR(EL) VRef) {}

size_t FN(EL, _Size)(__dpctl_keep VECTOR(EL) VRef) {}

size_t FN(EL, _GetAt)(__dpctl_keep VECTOR(EL) VRef, size_t index) {}

size_t FN(EL, _Get)(__dpctl_keep VECTOR(EL) VRef, size_t index) {}

void FN(EL, _PushBack)(__dpctl_keep VECTOR(EL) VRef,
                       __dpctl_take SYCLREF(EL) ELRef)
{
}