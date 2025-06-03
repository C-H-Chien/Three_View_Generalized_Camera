#ifndef magmaHC_dev_cuh_
#define magmaHC_dev_cuh_
// ============================================================================
// Device function for magma cgesv batched small
//
// Modifications
//    Chiang-Heng Chien  21-05-03:   Include device function from cgesv_batched_small.cu
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ============================================================================
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstring>

// -- magma included --
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

#include "definitions.hpp"
#include "typenames.hpp"

//namespace GPU_Device {

	template<int N>
    __device__ __inline__ void
	cgesv_batched_small_device(
		const int tx,
		magmaComplex rA[N], int* sipiv,
		magmaComplex &rB, magmaComplex *sB,
		magmaComplex *sx, float *dsx,
		int rowid, magma_int_t &linfo )
	{
		magmaComplex reg    = MAGMA_COMPLEX_ZERO;
		int max_id;

		#pragma unroll
		for(int i = 0; i < N; i++){
			FP_type rx_abs_max = MAGMA_NUM_ZERO;
			FP_type update = MAGMA_NUM_ZERO;
			// icamax and find pivot
			dsx[ rowid ] = fabs(MAGMA_COMPLEX_REAL( rA[i] )) + fabs(MAGMA_COMPLEX_IMAG( rA[i] ));
			magmablas_syncwarp();
			rx_abs_max = dsx[i];
			max_id = i;
			#pragma unroll
			for(int j = i+1; j < N; j++){
				if( dsx[j] > rx_abs_max){
					max_id = j;
					rx_abs_max = dsx[j];
				}
			}
			bool zero_pivot = (rx_abs_max == MAGMA_NUM_ZERO);
			linfo  = ( zero_pivot && linfo == 0) ? (i+1) : linfo;
			update = ( zero_pivot ) ? MAGMA_NUM_ZERO : MAGMA_NUM_ONE;

			if(rowid == max_id){
				sipiv[i] = max_id;
				rowid = i;
				#pragma unroll
				for(int j = i; j < N; j++){
					sx[j] = update * rA[j];
				}
				sB[0] = rB;
			}
			else if(rowid == i){
				rowid = max_id;
			}
			magmablas_syncwarp();

			reg = ( zero_pivot ) ? MAGMA_COMPLEX_ONE : MAGMA_COMPLEX_DIV(MAGMA_COMPLEX_ONE, sx[i] );
			// scal and ger
			if( rowid > i ){
				rA[i] *= reg;
				#pragma unroll
				for(int j = i+1; j < N; j++) {
					rA[j] -= rA[i] * sx[j];
				}
				rB -= rA[i] * sB[0];
			}
			magmablas_syncwarp();
		}

		sB[rowid] = rB;
		#pragma unroll
		for(int i = N-1; i >= 0; i--) {
			sx[rowid] = rA[i];
			magmablas_syncwarp();
			reg      = MAGMA_COMPLEX_DIV(sB[ i ], sx[ i ]);
			sB[ tx ] = (tx <  i) ? sB[ tx ] - reg * sx[ tx ]: sB[ tx ];
			sB[ tx ] = (tx == i) ? reg : sB[ tx ];
			magmablas_syncwarp();
		}
	}
//}

#endif
