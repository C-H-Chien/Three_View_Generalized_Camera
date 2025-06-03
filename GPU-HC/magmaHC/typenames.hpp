#ifndef TYPENAMES_HPP
#define TYPENAMES_HPP
// ============================================================================
// Defining typenames
//
// Modifications
//    Chiang-Heng Chien  25-05-31:      Enable single and double precision
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>

#include "magma_v2.h"
#include "definitions.hpp"

#if USE_SINGLE_PRECISION
typedef float                   FP_type;
typedef magmaFloatComplex       magmaComplex;
typedef magmaFloatComplex_ptr   magmaComplex_ptr;
#else
typedef double                  FP_type;
typedef magmaDoubleComplex      magmaComplex;
typedef magmaDoubleComplex_ptr  magmaComplex_ptr;
#endif

#endif
