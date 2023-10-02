#ifndef dev_eval_indxing_six_lines_16_cuh_
#define dev_eval_indxing_six_lines_16_cuh_
// ========================================================================================
// Device function for evaluating the parallel indexing for the Jacobians Hx, Ht, 
// and the homotopy H of six_lines_generalized_cam problem
//
// Modifications
//    Chiang-Heng Chien  22-11-23:   Initially created. Copied and edited from the 
//                                   dev-eval-indxing-six_lines_generalized_cam.cuh file.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ========================================================================================
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstring>

// -- cuda included --
#include <cuda_runtime.h>

// -- magma included --
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_internal.h"
#undef max
#undef min
#include "magma_templates.h"
#include "sync.cuh"
#undef max
#undef min
#include "shuffle.cuh"
#undef max
#undef min
#include "batched_kernel_param.h"

namespace magmaHCWrapper {

    //> compute the linear interpolations of parameters of phc
    template<int N, int N_3, int N_4>
    __device__ __inline__ void
    eval_parameter_homotopy(
        const int tx, float t, 
        magmaFloatComplex *s_phc_coeffs_Hx,
        magmaFloatComplex *s_phc_coeffs_Ht,
        const magmaFloatComplex __restrict__ *d_phc_coeffs_Hx,
        const magmaFloatComplex __restrict__ *d_phc_coeffs_Ht
    )
    {
        // =============================================================================
        //> parameter homotopy of Hx
        //> floor((156+1)/16) = 9
        #pragma unroll
        for (int i = 0; i < 9; i++) {
          s_phc_coeffs_Hx[ tx + i*N ] = d_phc_coeffs_Hx[ tx*4 + i*N_4 ] 
                                      + d_phc_coeffs_Hx[ tx*4 + 1 + i*N_4 ] * t
                                      + (d_phc_coeffs_Hx[ tx*4 + 2 + i*N_4 ] * t) * t
                                      + ((d_phc_coeffs_Hx[ tx*4 + 3 + i*N_4 ] * t) * t) * t;
        }

        // =============================================================================
        //> parameter homotopy of Ht
        //> floor((156+1)/16) = 9
        #pragma unroll
        for (int i = 0; i < 9; i++) {
          s_phc_coeffs_Ht[ tx + i*N ] = d_phc_coeffs_Ht[ tx*3 + i*N_3] 
                                      + d_phc_coeffs_Ht[ tx*3 + 1 + i*N_3 ] * t
                                      + (d_phc_coeffs_Ht[ tx*3 + 2 + i*N_3 ] * t) * t;
        }

        //> The remaining parts
        //> (156+1) - 16*9 = 157 - 144 = 13
        if (tx < 13) {
          //> parameter homotopy of Hx
          s_phc_coeffs_Hx[ tx + 144 ] = d_phc_coeffs_Hx[ tx*4 + 576 ] 
                                      + d_phc_coeffs_Hx[ tx*4 + 1 + 576 ] * t
                                      + (d_phc_coeffs_Hx[ tx*4 + 2 + 576 ] * t) * t
                                      + ((d_phc_coeffs_Hx[ tx*4 + 3 + 576 ] * t) * t) * t;
          
          //> parameter homotopy of Hx
          s_phc_coeffs_Ht[ tx + 144 ] = d_phc_coeffs_Ht[ tx*3 + 432 ] 
                                      + d_phc_coeffs_Ht[ tx*3 + 1 + 432 ] * t
                                      + (d_phc_coeffs_Ht[ tx*3 + 2 + 432 ] * t) * t;
        }
    }

    // -- Hx parallel indexing --
    template<int N, int max_terms, int max_parts, int max_terms_parts, int N_max_terms_parts>
    __device__ __inline__ void
    eval_Jacobian_Hx(
        const int tx, magmaFloatComplex *s_track, magmaFloatComplex r_cgesvA[N],
        const int* __restrict__ d_Hx_idx, magmaFloatComplex *s_phc_coeffs )
    {
      //#pragma unroll
      for(int i = 0; i < N; i++) {
        r_cgesvA[i] = MAGMA_C_ZERO;

        //#pragma unroll
        for(int j = 0; j < max_terms; j++) {
          r_cgesvA[i] += d_Hx_idx[j*max_parts + i*max_terms_parts + tx*N_max_terms_parts] 
                       * s_phc_coeffs[ d_Hx_idx[j*max_parts + 1 + i*max_terms_parts + tx*N_max_terms_parts] ]
                       * s_track[ d_Hx_idx[j*max_parts + 2 + i*max_terms_parts + tx*N_max_terms_parts] ];
        }
      }
    }

    // -- Ht parallel indexing --
    template<int N, int max_terms, int max_parts, int max_terms_parts>
    __device__ __inline__ void
    eval_Jacobian_Ht(
        const int tx, magmaFloatComplex *s_track, magmaFloatComplex &r_cgesvB,
        const int* __restrict__ d_Ht_idx, magmaFloatComplex *s_phc_coeffs)
    {
      r_cgesvB = MAGMA_C_ZERO;
      //#pragma unroll
      for (int i = 0; i < max_terms; i++) {
        r_cgesvB -= d_Ht_idx[i*max_parts + tx*max_terms_parts] 
                  * s_phc_coeffs[ d_Ht_idx[i*max_parts + 1 + tx*max_terms_parts] ]
                  * s_track[ d_Ht_idx[i*max_parts + 2 + tx*max_terms_parts] ] 
                  * s_track[ d_Ht_idx[i*max_parts + 3 + tx*max_terms_parts] ];
      }
    }

    // -- H parallel indexing --
    template<int N, int max_terms, int max_parts, int max_terms_parts>
    __device__ __inline__ void
    eval_homotopy(
        const int tx, magmaFloatComplex *s_track, magmaFloatComplex &r_cgesvB,
        const int* __restrict__ d_Ht_idx, magmaFloatComplex *s_phc_coeffs)
    {
      r_cgesvB = MAGMA_C_ZERO;
      //#pragma unroll
      for (int i = 0; i < max_terms; i++) {
        r_cgesvB += d_Ht_idx[i*max_parts + tx*max_terms_parts] 
                  * s_phc_coeffs[ d_Ht_idx[i*max_parts + 1 + tx*max_terms_parts] ]
                  * s_track[ d_Ht_idx[i*max_parts + 2 + tx*max_terms_parts] ] 
                  * s_track[ d_Ht_idx[i*max_parts + 3 + tx*max_terms_parts] ];
      }
    }
}

#endif
