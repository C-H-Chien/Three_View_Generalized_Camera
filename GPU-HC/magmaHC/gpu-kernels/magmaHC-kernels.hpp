#ifndef magmaHC_kernels_HPP
#define magmaHC_kernels_HPP
// ============================================================================
// Header file declaring all kernels
//
// Modifications
//    Chien  22-10-31:   Initially Created (Copied from other repos)
//    Chien  25-05-30:   Use the new format of CUDA kernels from the original GPU-HC code
//    Chien  25-06-03:   Adopt typenames that enable the use of both single and double precisions
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

#include "../typenames.hpp"
#include "magma_v2.h"

real_Double_t kernel_HC_Solver_generalized_3views_4pts(
  magma_queue_t         my_queue,
  int                   HC_max_steps, 
  int                   HC_max_correction_steps, 
  int                   HC_delta_t_incremental_steps,
  magmaComplex**   d_startSols_array, 
  magmaComplex**   d_Track_array,
  magma_int_t*          d_Hx_idx_array,
  magma_int_t*          d_Ht_idx_array, 
  magmaComplex_ptr d_phc_coeffs_Hx, 
  magmaComplex_ptr d_phc_coeffs_Ht,
  bool*                 d_is_GPU_HC_Sol_Converge,
  bool*                 d_is_GPU_HC_Sol_Infinity,
  magmaComplex*    d_Debug_Purpose
);

real_Double_t kernel_HC_Solver_generalized_3views_6lines(
  magma_queue_t         my_queue,
  int                   HC_max_steps, 
  int                   HC_max_correction_steps, 
  int                   HC_delta_t_incremental_steps,
  magmaComplex**   d_startSols_array, 
  magmaComplex**   d_Track_array,
  magma_int_t*          d_Hx_idx_array,
  magma_int_t*          d_Ht_idx_array, 
  magmaComplex_ptr d_phc_coeffs_Hx, 
  magmaComplex_ptr d_phc_coeffs_Ht,
  bool*                 d_is_GPU_HC_Sol_Converge,
  bool*                 d_is_GPU_HC_Sol_Infinity,
  magmaComplex*    d_Debug_Purpose
);

#endif
