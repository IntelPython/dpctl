#pragma once

#define xFN(TYPE, NAME) DPCTL##TYPE##Vector##_##NAME
#define FN(TYPE, NAME) xFN(TYPE, NAME)
#define xVECTOR(EL) DPCTL##EL##VectorRef
#define VECTOR(EL) xVECTOR(EL)
#define xSYCLREF(EL) DPCTLSycl##EL##Ref
#define SYCLREF(EL) xSYCLREF(EL)