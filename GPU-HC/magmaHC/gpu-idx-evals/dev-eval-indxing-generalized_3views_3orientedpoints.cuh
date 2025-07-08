#ifndef DEV_EVAL_INDXING_generalized_3views_3orientedpoints_CUH
#define DEV_EVAL_INDXING_generalized_3views_3orientedpoints_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
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

#include "../definitions.hpp"

//> compute the parameter homotopy
template < int Num_Of_Vars >
__device__ __inline__ void
compute_param_homotopy(
  const int tx, FP_type t,
  magmaComplex *s_start_params,
  magmaComplex *s_diff_params,
  magmaComplex *s_param_homotopy
)
{
	//> s_target_params now becomes parameter homotopy
	for (int i = 0; i < 3; i++) {
		// s_param_homotopy[ tx + i*Num_Of_Vars ] = s_target_params[ tx + i*Num_Of_Vars ] * t + s_start_params[ tx + i*Num_Of_Vars ] * (1.0-t);
		s_param_homotopy[ tx + i*Num_Of_Vars ] = s_diff_params[ tx + i*Num_Of_Vars ] * t + s_start_params[ tx + i*Num_Of_Vars ];
	}

    if (tx < 9) {
    //   s_param_homotopy[ tx + 45 ] = s_target_params[ tx + 45 ] * t + s_start_params[ tx + 45 ] * (1.0-t);
	  s_param_homotopy[ tx + 45 ] = s_diff_params[ tx + 45 ] * t + s_start_params[ tx + 45 ];
    }
}

//> compute coefficients for evaluating dH/dx and dH/dt
template < int Num_Of_Vars, int Num_Of_Terms_Per_Coeff, int Num_Of_P2C_Params_Per_Term, int Num_Of_Const_Coeffs_in_dHdt >
__device__ __inline__ void
eval_coefficients_for_Jacobians(
  const int tx, FP_type t,
  magmaComplex *s_diff_params,
  magmaComplex *s_param_homotopy,
  magmaComplex *s_dHdx_coeffs,
  magmaComplex *s_dHdt_coeffs,
  const int* __restrict__ d_P2C_indx
)
{
	//> Coefficients of dH/dx
	for (int i = 0; i < 22; i++) {
		s_dHdx_coeffs[i*Num_Of_Vars + tx] = MAGMA_MAKE_COMPLEX(0.0, 0.0);
		#pragma unroll 2
		for(int j = 0; j < Num_Of_Terms_Per_Coeff; j++) {
			int p_index = j*Num_Of_P2C_Params_Per_Term + (i*Num_Of_Vars + tx)*84;
			s_dHdx_coeffs[i*Num_Of_Vars + tx] += d_P2C_indx[p_index] * \
												 s_param_homotopy[ d_P2C_indx[p_index + 1] ] * \
												 s_param_homotopy[ d_P2C_indx[p_index + 2] ] * \
												 s_param_homotopy[ d_P2C_indx[p_index + 3] ] * \
												 s_param_homotopy[ d_P2C_indx[p_index + 4] ] * \
												 s_param_homotopy[ d_P2C_indx[p_index + 5] ] * \
												 s_param_homotopy[ d_P2C_indx[p_index + 6] ];
		}
	}

	//> Coefficients of dH/dt
	for (int i = 0; i < 22; i++) {
		s_dHdt_coeffs[i*Num_Of_Vars + tx] = MAGMA_MAKE_COMPLEX(0.0, 0.0);
		#pragma unroll 2
		for(int j = 0; j < Num_Of_Terms_Per_Coeff; j++) {
			int p_index = j*Num_Of_P2C_Params_Per_Term + (i*Num_Of_Vars + tx)*84;
			s_dHdt_coeffs[i*Num_Of_Vars + tx] += d_P2C_indx[p_index] * \
				(s_diff_params[ d_P2C_indx[p_index+1] ] * s_param_homotopy[ d_P2C_indx[p_index+2] ] * s_param_homotopy[ d_P2C_indx[p_index+3] ] * s_param_homotopy[ d_P2C_indx[p_index+4] ] * s_param_homotopy[ d_P2C_indx[p_index+5] ] * s_param_homotopy[ d_P2C_indx[p_index+6] ] \
			   + s_diff_params[ d_P2C_indx[p_index+2] ] * s_param_homotopy[ d_P2C_indx[p_index+1] ] * s_param_homotopy[ d_P2C_indx[p_index+3] ] * s_param_homotopy[ d_P2C_indx[p_index+4] ] * s_param_homotopy[ d_P2C_indx[p_index+5] ] * s_param_homotopy[ d_P2C_indx[p_index+6] ] \
			   + s_diff_params[ d_P2C_indx[p_index+3] ] * s_param_homotopy[ d_P2C_indx[p_index+1] ] * s_param_homotopy[ d_P2C_indx[p_index+2] ] * s_param_homotopy[ d_P2C_indx[p_index+4] ] * s_param_homotopy[ d_P2C_indx[p_index+5] ] * s_param_homotopy[ d_P2C_indx[p_index+6] ] \
			   + s_diff_params[ d_P2C_indx[p_index+4] ] * s_param_homotopy[ d_P2C_indx[p_index+1] ] * s_param_homotopy[ d_P2C_indx[p_index+2] ] * s_param_homotopy[ d_P2C_indx[p_index+3] ] * s_param_homotopy[ d_P2C_indx[p_index+5] ] * s_param_homotopy[ d_P2C_indx[p_index+6] ] \
			   + s_diff_params[ d_P2C_indx[p_index+5] ] * s_param_homotopy[ d_P2C_indx[p_index+1] ] * s_param_homotopy[ d_P2C_indx[p_index+2] ] * s_param_homotopy[ d_P2C_indx[p_index+3] ] * s_param_homotopy[ d_P2C_indx[p_index+4] ] * s_param_homotopy[ d_P2C_indx[p_index+6] ] \
			   + s_diff_params[ d_P2C_indx[p_index+6] ] * s_param_homotopy[ d_P2C_indx[p_index+1] ] * s_param_homotopy[ d_P2C_indx[p_index+2] ] * s_param_homotopy[ d_P2C_indx[p_index+3] ] * s_param_homotopy[ d_P2C_indx[p_index+4] ] * s_param_homotopy[ d_P2C_indx[p_index+5] ]);
		}
	}
}

template< int Num_Of_Vars, int dHdx_Max_Terms, int dHdx_Max_Parts, int dHdx_Entry_Offset, int dHdx_Row_Offset >
__device__ __inline__ void
eval_Jacobian_Hx(
	 const int tx, magmaComplex *s_track, magmaComplex r_cgesvA[Num_Of_Vars],
	 const int* __restrict__ d_Hx_idx, magmaComplex *s_phc_coeffs )
{
	for(int i = 0; i < Num_Of_Vars; i++) {
		r_cgesvA[i] = MAGMA_C_ZERO;

		#pragma unroll 2
		for(int j = 0; j < dHdx_Max_Terms; j++) {
			r_cgesvA[i] += d_Hx_idx[j*dHdx_Max_Parts + i*dHdx_Entry_Offset + tx*dHdx_Row_Offset] 
						* s_phc_coeffs[ d_Hx_idx[j*dHdx_Max_Parts + 1 + i*dHdx_Entry_Offset + tx*dHdx_Row_Offset] ]
						* s_track[      d_Hx_idx[j*dHdx_Max_Parts + 2 + i*dHdx_Entry_Offset + tx*dHdx_Row_Offset] ]
						* s_track[      d_Hx_idx[j*dHdx_Max_Parts + 3 + i*dHdx_Entry_Offset + tx*dHdx_Row_Offset] ]
						* s_track[      d_Hx_idx[j*dHdx_Max_Parts + 4 + i*dHdx_Entry_Offset + tx*dHdx_Row_Offset] ];
		}
	}
}

template< int dHdt_Max_Terms, int dHdt_Max_Parts, int dHdt_Row_Offset >
__device__ __inline__ void
eval_Jacobian_Ht(
	 const int tx, magmaComplex *s_track, magmaComplex &r_cgesvB,
	 const int* __restrict__ d_Ht_idx, magmaComplex *s_phc_coeffs )
{
		 r_cgesvB = MAGMA_C_ZERO;
		 #pragma unroll 2
		 for (int i = 0; i < dHdt_Max_Terms; i++) {
			 r_cgesvB -= d_Ht_idx[i*dHdt_Max_Parts + tx*dHdt_Row_Offset]
			              * s_phc_coeffs[ d_Ht_idx[i*dHdt_Max_Parts + 1 + tx*dHdt_Row_Offset] ]
				 		  * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 2 + tx*dHdt_Row_Offset] ]
				 		  * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 3 + tx*dHdt_Row_Offset] ]
				 		  * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 4 + tx*dHdt_Row_Offset] ]
				 		  * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 5 + tx*dHdt_Row_Offset] ];
		 }
}

template< int dHdt_Max_Terms, int dHdt_Max_Parts, int dHdt_Row_Offset >
__device__ __inline__ void
eval_Homotopy(
	 const int tx, magmaComplex *s_track, magmaComplex &r_cgesvB,
	 const int* __restrict__ d_Ht_idx, magmaComplex *s_phc_coeffs)
{
		 r_cgesvB = MAGMA_C_ZERO;
		 #pragma unroll 2
		 for (int i = 0; i < dHdt_Max_Terms; i++) {
			 r_cgesvB += d_Ht_idx[i*dHdt_Max_Parts + tx*dHdt_Row_Offset] 
			 * s_phc_coeffs[ d_Ht_idx[i*dHdt_Max_Parts + 1 + tx*dHdt_Row_Offset] ] 
			 * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 2 + tx*dHdt_Row_Offset] ]
			 * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 3 + tx*dHdt_Row_Offset] ]
			 * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 4 + tx*dHdt_Row_Offset] ]
			 * s_track[      d_Ht_idx[i*dHdt_Max_Parts + 5 + tx*dHdt_Row_Offset] ];
		 }
}

#endif