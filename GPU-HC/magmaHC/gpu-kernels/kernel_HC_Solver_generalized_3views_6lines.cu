#ifndef KERNEL_HC_solver_generalized_3views_6lines_CU
#define KERNEL_HC_solver_generalized_3views_6lines_CU
// =============================================================================================================
// GPU homotopy continuation solver for Generalized Three-View Relative Pose Problem from 6 lines
//
// Modifications
//    Chiang-Heng Chien  24-05-25:   Build on top of generalized 3-views from 4-points
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =============================================================================================================
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

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
#include "magmaHC-kernels.hpp"

//> device functions
#include "../gpu-idx-evals/dev-eval-indxing-generalized_3views_6lines.cuh"
#include "../dev-cgesv-batched-small.cuh"

template< int Num_Of_Vars, int Num_of_Coeffs_from_Params, int Max_Order_of_t, \
          int dHdx_Max_Terms, int dHdx_Max_Parts, int dHdx_Entry_Offset, int dHdx_Row_Offset, \
          int dHdt_Max_Terms, int dHdt_Max_Parts, int dHdt_Row_Offset, \
          unsigned Full_Parallel_Offset, \
          unsigned Partial_Parallel_Thread_Offset, \
          unsigned Partial_Parallel_Index_Offset, \
          unsigned Max_Order_of_t_Plus_One, \
          unsigned Partial_Parallel_Index_Offset_Hx, \
          unsigned Partial_Parallel_Index_Offset_Ht >
__global__ void
HC_solver_generalized_3views_6lines(
  int HC_max_steps, int HC_max_correction_steps, int HC_delta_t_incremental_steps,
  magmaComplex** d_startSols_array, magmaComplex** d_Track_array,
  magma_int_t* d_Hx_indices, magma_int_t* d_Ht_indices,
  magmaComplex_ptr d_phc_coeffs_Hx, magmaComplex_ptr d_phc_coeffs_Ht,
  bool* d_is_GPU_HC_Sol_Converge, bool* d_is_GPU_HC_Sol_Infinity,
  magmaComplex* d_Debug_Purpose
)
{
  const int tx = threadIdx.x;
  const int batchid = blockIdx.x ;

  magmaComplex* d_startSols   = d_startSols_array[batchid];
  magmaComplex* d_track       = d_Track_array[batchid];
  const int* __restrict__ d_Hx_idx = d_Hx_indices;
  const int* __restrict__ d_Ht_idx = d_Ht_indices;
  const magmaComplex* __restrict__ d_const_phc_coeffs_Hx = d_phc_coeffs_Hx;
  const magmaComplex* __restrict__ d_const_phc_coeffs_Ht = d_phc_coeffs_Ht;

  //> registers declarations
  magmaComplex r_cgesvA[Num_Of_Vars] = {MAGMA_COMPLEX_ZERO};
  magmaComplex r_cgesvB = MAGMA_COMPLEX_ZERO;
  int linfo = 0, rowid = tx;
  FP_type t0 = 0.0, t_step = 0.0, delta_t = 0.05;
  bool end_zone = 0;

  //> shared memory declarations
#if USE_SINGLE_PRECISION
  extern __shared__ magmaComplex zdata[];
  magmaComplex *s_sols               = (magmaComplex*)(zdata);
  magmaComplex *s_track              = s_sols                   + (Num_Of_Vars+1);
  magmaComplex *s_track_last_success = s_track                  + (Num_Of_Vars+1);
  magmaComplex *sB                   = s_track_last_success     + (Num_Of_Vars+1);
  magmaComplex *sx                   = sB                       + Num_Of_Vars;
  magmaComplex *s_phc_coeffs_Hx      = sx                       + Num_Of_Vars;
  magmaComplex *s_phc_coeffs_Ht      = s_phc_coeffs_Hx          + (Num_of_Coeffs_from_Params+1);
  FP_type* dsx                       = (FP_type*)(s_phc_coeffs_Ht + (Num_of_Coeffs_from_Params+1));
  int* sipiv                         = (int*)(dsx               + Num_Of_Vars);
  FP_type* s_delta_t_scale           = (FP_type*)(sipiv         + (Num_Of_Vars+1));
  int* s_RK_Coeffs                   = (int*)(s_delta_t_scale   + 1);
#else
  //> make sure that the memory address is well-aligned when using double precision
  extern __shared__ char shared_mem[];
  SHARED_ALLOC_INIT(shared_mem);  // set up internal pointer
  SHARED_ALLOC(magmaComplex, s_sols,  Num_Of_Vars+1);
  SHARED_ALLOC(magmaComplex, s_track, Num_Of_Vars+1);
  SHARED_ALLOC(magmaComplex, s_track_last_success, Num_Of_Vars+1);
  SHARED_ALLOC(magmaComplex, sB, Num_Of_Vars);
  SHARED_ALLOC(magmaComplex, sx, Num_Of_Vars);
  SHARED_ALLOC(magmaComplex, s_phc_coeffs_Hx, Num_of_Coeffs_from_Params+1);
  SHARED_ALLOC(magmaComplex, s_phc_coeffs_Ht, Num_of_Coeffs_from_Params+1);
  SHARED_ALLOC(FP_type, dsx, Num_Of_Vars);
  SHARED_ALLOC(int, sipiv, Num_Of_Vars+1);
  SHARED_ALLOC(FP_type, s_delta_t_scale, 1);
  SHARED_ALLOC(int, s_RK_Coeffs, 1); 
#endif

  s_sols[tx] = d_startSols[tx];
  s_track[tx] = d_track[tx];
  s_track_last_success[tx] = s_track[tx];
  if (tx == 0) {
    s_sols[Num_Of_Vars]               = MAGMA_MAKE_COMPLEX(1.0, 0.0);
    s_track[Num_Of_Vars]              = MAGMA_MAKE_COMPLEX(1.0, 0.0);
    s_track_last_success[Num_Of_Vars] = MAGMA_MAKE_COMPLEX(1.0, 0.0);
    sipiv[Num_Of_Vars]                = 0;
  }
  magmablas_syncwarp();

  FP_type one_half_delta_t;   //> 1/2 \Delta t
  FP_type r_sqrt_sols;
  FP_type r_sqrt_corr;
  bool r_isSuccessful;
  bool r_isInfFail;

  //#pragma unroll
  volatile int hc_max_steps = HC_max_steps;
  for (int step = 0; step <= hc_max_steps; step++) {
    if (t0 < 1.0 && (1.0-t0 > 0.0000001)) {

      // ===================================================================
      //> Decide delta t at end zone
      // ===================================================================
      if (!end_zone && fabs(1 - t0) <= (0.0500001)) {
        end_zone = true;
      }

      if (end_zone) {
        if (delta_t > fabs(1 - t0))
          delta_t = fabs(1 - t0);
      }
      else if (delta_t > fabs(1 - 0.05 - t0)) {
        delta_t = fabs(1 - 0.05 - t0);
      }

      t_step = t0;
      one_half_delta_t = 0.5 * delta_t;

      // ===================================================================
      //> Runge-Kutta Predictor
      // ===================================================================
      unsigned char scales[3] = {1, 0, 1};
      if (tx == 0) {
        s_delta_t_scale[0] = 0.0;
        s_RK_Coeffs[0] = 1;
      }
      magmablas_syncwarp();

      //> For simplicity, let's stay with no gamma-trick mode
      #pragma no unroll
      for (int rk_step = 0; rk_step < 4; rk_step++ ) {

        //> Evaluate parameter homotopy
        eval_parameter_homotopy< Num_Of_Vars, Max_Order_of_t, Full_Parallel_Offset, Partial_Parallel_Thread_Offset, Partial_Parallel_Index_Offset, \
                                Max_Order_of_t_Plus_One, Partial_Parallel_Index_Offset_Hx, Partial_Parallel_Index_Offset_Ht > \
                                ( tx, t0, s_phc_coeffs_Hx, s_phc_coeffs_Ht, d_const_phc_coeffs_Hx, d_const_phc_coeffs_Ht );

        //> Evaluate dH/dx and dH/dt
        eval_Jacobian_Hx< Num_Of_Vars, dHdx_Max_Terms, dHdx_Max_Parts, dHdx_Entry_Offset, dHdx_Row_Offset >( tx, s_track, r_cgesvA, d_Hx_idx, s_phc_coeffs_Hx );
        eval_Jacobian_Ht< dHdt_Max_Terms, dHdt_Max_Parts, dHdt_Row_Offset >( tx, s_track, r_cgesvB, d_Ht_idx, s_phc_coeffs_Ht );

        //> linear system solver: solve for k1, k2, k3, or k4
        cgesv_batched_small_device< Num_Of_Vars >( tx, r_cgesvA, sipiv, r_cgesvB, sB, sx, dsx, rowid, linfo );
        magmablas_syncwarp();

        if (rk_step < 3) {

          s_sols[tx] += sB[tx] * delta_t * (s_RK_Coeffs[0] * 1.0/6.0);
          s_track[tx] = (s_RK_Coeffs[0] > 1) ? s_track_last_success[tx] : s_track[tx];

          if (tx == 0) {
            s_delta_t_scale[0] += scales[rk_step] * one_half_delta_t;
            s_RK_Coeffs[0] = s_RK_Coeffs[0] << scales[rk_step];           //> Shift one bit
          }
          magmablas_syncwarp();

          sB[tx] *= s_delta_t_scale[0];
          s_track[tx] += sB[tx];
          t0 += scales[rk_step] * one_half_delta_t;
        }
        magmablas_syncwarp();
      }
      //> Make prediction
      s_sols[tx] += sB[tx] * delta_t * 1.0/6.0;
      s_track[tx] = s_sols[tx];
      magmablas_syncwarp();

      // ===================================================================
      //> Gauss-Newton Corrector
      // ===================================================================
      //#pragma unroll
      volatile int hc_max_xorrection_steps = HC_max_correction_steps;
      #pragma no unroll
      for(int i = 0; i < hc_max_xorrection_steps; i++) {

        eval_Jacobian_Hx< Num_Of_Vars, dHdx_Max_Terms, dHdx_Max_Parts, dHdx_Entry_Offset, dHdx_Row_Offset >( tx, s_track, r_cgesvA, d_Hx_idx, s_phc_coeffs_Hx );
        eval_Homotopy< dHdt_Max_Terms, dHdt_Max_Parts, dHdt_Row_Offset >( tx, s_track, r_cgesvB, d_Ht_idx, s_phc_coeffs_Hx );

        //> G-N corrector first solve
        cgesv_batched_small_device< Num_Of_Vars >( tx, r_cgesvA, sipiv, r_cgesvB, sB, sx, dsx, rowid, linfo );
        magmablas_syncwarp();

        //> correct the sols
        s_track[tx] -= sB[tx];
        magmablas_syncwarp();

        r_sqrt_sols = MAGMA_COMPLEX_REAL(sB[tx])*MAGMA_COMPLEX_REAL(sB[tx]) + MAGMA_COMPLEX_IMAG(sB[tx])*MAGMA_COMPLEX_IMAG(sB[tx]);
        r_sqrt_corr = MAGMA_COMPLEX_REAL(s_track[tx])*MAGMA_COMPLEX_REAL(s_track[tx]) + MAGMA_COMPLEX_IMAG(s_track[tx])*MAGMA_COMPLEX_IMAG(s_track[tx]);
        magmablas_syncwarp();

        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2 ) {
            r_sqrt_sols += __shfl_down_sync(__activemask(), r_sqrt_sols, offset);
            r_sqrt_corr += __shfl_down_sync(__activemask(), r_sqrt_corr, offset);
        }

        if ( tx == 0 ) {
            r_isSuccessful = r_sqrt_sols < 0.000001 * r_sqrt_corr;
            r_isInfFail = (r_sqrt_corr > 1e14) ? (true) : (false);
        }
        //> Broadcast the values of r_isSuccessful and r_isInfFail from thread 0 to all the rest of the threads
        r_isSuccessful = __shfl_sync(__activemask(), r_isSuccessful, 0);
        r_isInfFail = __shfl_sync(__activemask(), r_isInfFail, 0);

        if (r_isInfFail) break;
        if (r_isSuccessful) break;
      }

      if ( r_isInfFail ) break;

      // ===================================================================
      //> Decide Track Changes
      // ===================================================================
      if (!r_isSuccessful) {
        delta_t *= 0.5;
        //> should be the last successful tracked sols
        s_track[tx] = s_track_last_success[tx];
        s_sols[tx] = s_track_last_success[tx];
        if (tx == 0) sipiv[Num_Of_Vars] = 0;
        magmablas_syncwarp();
        t0 = t_step;
      }
      else {
        if (tx == 0) sipiv[Num_Of_Vars]++;
        s_track_last_success[tx] = s_track[tx];
        s_sols[tx] = s_track[tx];
        magmablas_syncwarp();
        if (sipiv[Num_Of_Vars] >= HC_delta_t_incremental_steps) {
          if (tx == 0) sipiv[Num_Of_Vars] = 0;
          delta_t *= 2;
        }
        magmablas_syncwarp();
      }
    }
    else {
      break;
    }
  }

  d_track[tx] = s_track[tx];
  if (tx == 0) {
    d_is_GPU_HC_Sol_Converge[ batchid ] = (t0 >= 1.0 || (1.0-t0 <= 0.0000001)) ? (1) : (0);
    d_is_GPU_HC_Sol_Infinity[ batchid ] = (r_isInfFail) ? (1) : (0);
  }

#if GPU_DEBUG
  d_Debug_Purpose[ batchid ] = (t0 >= 1.0 || (1.0-t0 <= 0.0000001)) ? MAGMA_MAKE_COMPLEX(1.0, 0.0) : MAGMA_MAKE_COMPLEX(t0, delta_t);
#endif
}

real_Double_t
kernel_HC_Solver_generalized_3views_6lines(
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
)
{
  //> Hard-coded for each problem
  const int num_of_vars               = 6;
  const int num_of_tracks             = 600;
  const int num_of_coeffs_from_params = 564;
  const int dHdx_Max_Terms            = 40;
  const int dHdx_Max_Parts            = 5;
  const int dHdt_Max_Terms            = 100;
  const int dHdt_Max_Parts            = 6;
  const int max_order_of_t            = 3;

  const int dHdx_Entry_Offset = dHdx_Max_Terms * dHdx_Max_Parts;
  const int dHdx_Row_Offset   = num_of_vars * dHdx_Entry_Offset;
  const int dHdt_Row_Offset   = dHdt_Max_Terms * dHdt_Max_Parts;

  real_Double_t gpu_time;
  dim3 threads(num_of_vars, 1, 1);
  dim3 grid(num_of_tracks, 1, 1);
  cudaError_t e = cudaErrorInvalidValue;

  //> Constant values for evaluating the Jacobians, passed as template
  const unsigned Full_Parallel_Offset                 = (num_of_coeffs_from_params+1)/(num_of_vars);
  const unsigned Partial_Parallel_Thread_Offset       = (num_of_coeffs_from_params+1) - (num_of_vars)*(Full_Parallel_Offset);
  const unsigned Partial_Parallel_Index_Offset        = (num_of_vars)*(Full_Parallel_Offset);
  const unsigned Max_Order_of_t_Plus_One              = max_order_of_t + 1;
  const unsigned Partial_Parallel_Index_Offset_for_Hx = (num_of_vars-1)*(Max_Order_of_t_Plus_One) + (max_order_of_t) + (Full_Parallel_Offset-1)*(Max_Order_of_t_Plus_One)*(num_of_vars) + 1;
  const unsigned Partial_Parallel_Index_Offset_for_Ht = (num_of_vars-1)*(max_order_of_t) + (max_order_of_t-1) + (Full_Parallel_Offset-1)*(max_order_of_t)*(num_of_vars) + 1;

  //> declare shared memory
  // magma_int_t shmem  = 0;
  // shmem += (num_of_vars+1)               * sizeof(magmaComplex);   // startSols
  // shmem += (num_of_vars+1)               * sizeof(magmaComplex);   // track
  // shmem += (num_of_vars+1)               * sizeof(magmaComplex);   // track_pred_init
  // shmem += (num_of_coeffs_from_params+1) * sizeof(magmaComplex);   // s_phc_coeffs_Hx
  // shmem += (num_of_coeffs_from_params+1) * sizeof(magmaComplex);   // s_phc_coeffs_Ht
  // shmem += num_of_vars                   * sizeof(magmaComplex);   // sB
  // shmem += num_of_vars                   * sizeof(magmaComplex);   // sx
  // shmem += num_of_vars                   * sizeof(FP_type);               // dsx
  // shmem += num_of_vars                   * sizeof(int);                 // pivot
  // shmem += 1 * sizeof(int);                                             // predictor_success counter
  // shmem += 1 * sizeof(int);                                             // Loopy Runge-Kutta coefficients
  // shmem += 1 * sizeof(FP_type);                                           // Loopy Runge-Kutta delta t

  magma_int_t shmem  = 0;
#if USE_SINGLE_PRECISION
  shmem += (num_of_vars+1)               * sizeof(magmaComplex);   // s_sols
  shmem += (num_of_vars+1)               * sizeof(magmaComplex);   // s_track
  shmem += (num_of_vars+1)               * sizeof(magmaComplex);   // s_track_last_success
  shmem += num_of_vars                   * sizeof(magmaComplex);   // sB
  shmem += num_of_vars                   * sizeof(magmaComplex);   // sx
  shmem += (num_of_coeffs_from_params+1) * sizeof(magmaComplex);   // s_phc_coeffs_Hx
  shmem += (num_of_coeffs_from_params+1) * sizeof(magmaComplex);   // s_phc_coeffs_Ht
  shmem += num_of_vars                   * sizeof(FP_type);               // dsx
  shmem += (num_of_vars+1)               * sizeof(int);                 // pivot
  shmem += 1 * sizeof(FP_type);                                           // s_delta_t_scale
  shmem += 1 * sizeof(int);                                             // s_RK_Coeffs
#else
  auto align = [](size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
  };
  shmem = align(shmem, alignof(magmaComplex));
  shmem += sizeof(magmaComplex) * (num_of_vars+1);  // s_sols
  shmem = align(shmem, alignof(magmaComplex));
  shmem += sizeof(magmaComplex) * (num_of_vars+1);  // s_track
  shmem = align(shmem, alignof(magmaComplex));
  shmem += sizeof(magmaComplex) * (num_of_vars+1);  // s_track_last_success
  shmem = align(shmem, alignof(magmaComplex));
  shmem += sizeof(magmaComplex) * (num_of_vars);
  shmem = align(shmem, alignof(magmaComplex));
  shmem += sizeof(magmaComplex) * (num_of_vars);
  shmem = align(shmem, alignof(magmaComplex));
  shmem += sizeof(magmaComplex) * (num_of_coeffs_from_params+1);
  shmem = align(shmem, alignof(magmaComplex));
  shmem += sizeof(magmaComplex) * (num_of_coeffs_from_params+1);
  shmem = align(shmem, alignof(FP_type));
  shmem += sizeof(FP_type) * num_of_vars;
  shmem = align(shmem, alignof(FP_type));
  shmem += sizeof(FP_type) * 1;
  shmem = align(shmem, alignof(int));
  shmem += sizeof(int) * (num_of_vars+1);
  shmem = align(shmem, alignof(int));
  shmem += sizeof(int) * 1;
#endif 

  //> Get max. dynamic shared memory on the GPU
  int nthreads_max, shmem_max = 0;
  cudacheck( cudaDeviceGetAttribute(&nthreads_max, cudaDevAttrMaxThreadsPerBlock, 0) );
#if CUDA_VERSION >= 9000
  cudacheck( cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0) );
  if (shmem <= shmem_max) {
    cudacheck( cudaFuncSetAttribute(HC_solver_generalized_3views_6lines \
                                    < num_of_vars, num_of_coeffs_from_params, max_order_of_t, \
                                      dHdx_Max_Terms, dHdx_Max_Parts, dHdx_Entry_Offset, dHdx_Row_Offset, \
                                      dHdt_Max_Terms, dHdt_Max_Parts, dHdt_Row_Offset, \
                                      Full_Parallel_Offset, \
                                      Partial_Parallel_Thread_Offset, \
                                      Partial_Parallel_Index_Offset, \
                                      Max_Order_of_t_Plus_One, \
                                      Partial_Parallel_Index_Offset_for_Hx, \
                                      Partial_Parallel_Index_Offset_for_Ht>, \
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, shmem) );
  }
#else
  cudacheck( cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, 0) );
#endif

  //> Message of overuse shared memory
  if ( shmem > shmem_max ) printf("Error: kernel %s requires too many threads or too much shared memory\n", __func__);

  void *kernel_args[] = { &HC_max_steps, &HC_max_correction_steps, &HC_delta_t_incremental_steps, \
                          &d_startSols_array, &d_Track_array, \
                          &d_Hx_idx_array, &d_Ht_idx_array, \
                          &d_phc_coeffs_Hx, &d_phc_coeffs_Ht, \
                          &d_is_GPU_HC_Sol_Converge, &d_is_GPU_HC_Sol_Infinity, \
                          &d_Debug_Purpose };

  gpu_time = magma_sync_wtime( my_queue );

  // cudacheck( cudaEventRecord(start) );
  e = cudaLaunchKernel((void*)HC_solver_generalized_3views_6lines \
                        < num_of_vars, num_of_coeffs_from_params, max_order_of_t, \
                          dHdx_Max_Terms, dHdx_Max_Parts, dHdx_Entry_Offset, dHdx_Row_Offset, \
                          dHdt_Max_Terms, dHdt_Max_Parts, dHdt_Row_Offset, \
                          Full_Parallel_Offset, \
                          Partial_Parallel_Thread_Offset, \
                          Partial_Parallel_Index_Offset, \
                          Max_Order_of_t_Plus_One, \
                          Partial_Parallel_Index_Offset_for_Hx, \
                          Partial_Parallel_Index_Offset_for_Ht>, \
                        grid, threads, kernel_args, shmem, my_queue->cuda_stream());

  gpu_time = magma_sync_wtime( my_queue ) - gpu_time;
  if( e != cudaSuccess ) printf("cudaLaunchKernel of HC_solver_generalized_3views_6lines is not successful!\n");

  return gpu_time;
}

#endif
