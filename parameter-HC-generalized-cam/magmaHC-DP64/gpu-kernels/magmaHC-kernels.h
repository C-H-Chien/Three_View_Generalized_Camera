#ifndef magmaHC_kernels_h
#define magmaHC_kernels_h
// ============================================================================
// Header file declaring all kernels
//
// Modifications
//    Chiang-Heng Chien  22-10-31:   Initially Created (Copied from other repos)
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>

#include "../magmaHC-problems.cuh"

// -- magma --
#include "flops.h"
#include "magma_v2.h"

extern "C" {
namespace magmaHCWrapperDP64 {

  real_Double_t kernel_HC_Solver_DP64_six_lines_6x6(
    magma_int_t N, magma_int_t batchCount, magma_int_t ldda,
    magma_queue_t my_queue,
    magmaDoubleComplex** d_startSols_array, magmaDoubleComplex** d_Track_array,
    magmaDoubleComplex** d_cgesvA_array, magmaDoubleComplex** d_cgesvB_array,
    magma_int_t* d_Hx_idx_array, magma_int_t* d_Ht_idx_array,
    magmaDoubleComplex_ptr d_phc_coeffs_Hx, magmaDoubleComplex_ptr d_phc_coeffs_Ht,
    magma_int_t numOf_phc_coeffs
  );

  real_Double_t kernel_HC_Solver_DP64_3views_4pts(
    magma_int_t N, magma_int_t batchCount, magma_int_t ldda,
    magma_queue_t my_queue,
    magmaDoubleComplex** d_startSols_array, magmaDoubleComplex** d_Track_array,
    magmaDoubleComplex** d_cgesvA_array, magmaDoubleComplex** d_cgesvB_array,
    magma_int_t* d_Hx_idx_array, magma_int_t* d_Ht_idx_array,
    magmaDoubleComplex_ptr d_phc_coeffs_Hx, magmaDoubleComplex_ptr d_phc_coeffs_Ht,
    magma_int_t numOf_phc_coeffs
  );

}
}

#endif
