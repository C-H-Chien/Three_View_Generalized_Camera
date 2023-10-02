#ifndef homotopy_continuation_solver_cu
#define homotopy_continuation_solver_cu
// =========================================================================================
//
// Modifications
//    Chiang-Heng Chien  22-10-18:   Initially Created (Copied from other repos)
//    Chiang-Heng Chien  23-02-26:   Chang single precision to double precision
//
// Notes
//    Chiang-Heng Chien  22-11-12:   Some parts of this script should be reorganized a bit.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ==========================================================================================
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <vector>

// cuda included
#include <cuda.h>
#include <cuda_runtime.h>

// magma
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

#include "magmaHC-problems.cuh"
#include "gpu-kernels/magmaHC-kernels.h"

namespace magmaHCWrapperDP64 {

  void homotopy_continuation_solver(
    magmaDoubleComplex *h_startSols, magmaDoubleComplex *h_Track,
    magmaDoubleComplex *h_startParams, magmaDoubleComplex *h_targetParams,
    magma_int_t *h_Hx_idx, magma_int_t *h_Ht_idx,
    magmaDoubleComplex *h_phc_coeffs_Hx, magmaDoubleComplex *h_phc_coeffs_Ht,
    problem_params* pp, std::string hc_problem, std::ofstream &track_sols_file,
    std::ofstream &timings_file
    )
  {
    magma_init();
    magma_print_environment();

    magma_int_t batchCount = pp->numOfTracks;
    magma_int_t coefsCount = pp->numOfParams;
    magma_int_t N = pp->numOfVars;

    real_Double_t     gpu_time;
    real_Double_t     data_h2d_time, data_d2h_time;
    magmaDoubleComplex *h_cgesvA, *h_cgesvB;
    magmaDoubleComplex *h_cgesvA_verify, *h_cgesvB_verify;
    magmaDoubleComplex *h_track_sols;
    magmaDoubleComplex_ptr d_startSols, d_Track;
    magmaDoubleComplex_ptr d_startCoefs, d_targetCoefs;
    magmaDoubleComplex_ptr d_cgesvA, d_cgesvB;
    magma_int_t lda, ldb, ldda, lddb, ldd_coefs, sizeA, sizeB;
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magmaDoubleComplex **d_startSols_array = NULL;
    magmaDoubleComplex **d_Track_array = NULL;
    magmaDoubleComplex **d_startCoefs_array = NULL;
    magmaDoubleComplex **d_targetCoefs_array = NULL;
    magmaDoubleComplex **d_cgesvA_array = NULL;
    magmaDoubleComplex **d_cgesvB_array = NULL;

    magma_device_t cdev;       // variable to indicate current gpu id
    magma_queue_t my_queue;    // magma queue variable, internally holds a cuda stream and a cublas handle
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &my_queue );     // create a queue on this cdev

    lda    = N;
    ldb    = lda;
    ldda   = magma_roundup( N, 32 );  // multiple of 32 by default
    lddb   = ldda;
    ldd_coefs = magma_roundup( coefsCount, 32 );  // multiple of 32 by default
    sizeA = lda*N*batchCount;
    sizeB = ldb*batchCount;

    // ==================================================================================================
    // -- Hx and Ht index matrices --
    // ==================================================================================================
    magma_int_t *d_Hx_idx;
    magma_int_t *d_Ht_idx;
    magma_int_t size_Hx = N*N*pp->Hx_maximal_terms*(pp->Hx_maximal_parts);
    magma_int_t size_Ht = N*pp->Ht_maximal_terms*(pp->Ht_maximal_parts);
  
    // -- allocate gpu memories --
    magma_imalloc( &d_Hx_idx, size_Hx );
    magma_imalloc( &d_Ht_idx, size_Ht );
    
    // -- transfer data from cpu to gpu --
    magma_isetmatrix( size_Hx, 1, h_Hx_idx, size_Hx, d_Hx_idx, size_Hx, my_queue );
    magma_isetmatrix( size_Ht, 1, h_Ht_idx, size_Ht, d_Ht_idx, size_Ht, my_queue );
    // ==================================================================================================

    // ==================================================================================================
    // -- coefficient p2c of funtion t for Hx and Ht --
    // ==================================================================================================
    magma_int_t phc_coeffs_Hx_size = (pp->numOfCoeffsFromParams+1)*(pp->max_orderOf_t+1);
    magma_int_t phc_coeffs_Ht_size = (pp->numOfCoeffsFromParams+1)*(pp->max_orderOf_t);
    magma_int_t ldd_phc_coeffs_Hx = magma_roundup( phc_coeffs_Hx_size, 32 );  // multiple of 32 by default
    magma_int_t ldd_phc_coeffs_Ht = magma_roundup( phc_coeffs_Ht_size, 32 );  // multiple of 32 by default
    magmaDoubleComplex_ptr d_phc_coeffs_Hx;
    magmaDoubleComplex_ptr d_phc_coeffs_Ht;
    // -- allocate GPU memory --
    magma_zmalloc( &d_phc_coeffs_Hx, ldd_phc_coeffs_Hx );
    magma_zmalloc( &d_phc_coeffs_Ht, ldd_phc_coeffs_Ht );
    // -- transfer from CPU to GPU --
    magma_zsetmatrix( phc_coeffs_Hx_size, 1, h_phc_coeffs_Hx, phc_coeffs_Hx_size, d_phc_coeffs_Hx, ldd_phc_coeffs_Hx, my_queue );
    magma_zsetmatrix( phc_coeffs_Ht_size, 1, h_phc_coeffs_Ht, phc_coeffs_Ht_size, d_phc_coeffs_Ht, ldd_phc_coeffs_Ht, my_queue );
    // ==================================================================================================

    // -- allocate CPU memory --
    magma_zmalloc_cpu( &h_cgesvA, N*N*batchCount );
    magma_zmalloc_cpu( &h_cgesvB, N*batchCount );
    magma_zmalloc_cpu( &h_cgesvA_verify, N*N*batchCount );
    magma_zmalloc_cpu( &h_cgesvB_verify, N*batchCount );
    magma_zmalloc_cpu( &h_track_sols, (N+1)*batchCount );

    // -- allocate GPU gm --
    magma_zmalloc( &d_startSols, (N+1)*batchCount );
    magma_zmalloc( &d_Track, (N+1)*batchCount );
    magma_zmalloc( &d_startCoefs, ldd_coefs );
    magma_zmalloc( &d_targetCoefs, ldd_coefs );
    magma_zmalloc( &d_cgesvA, ldda*N*batchCount );
    magma_zmalloc( &d_cgesvB, ldda*batchCount );

    // -- allocate 2d arrays in GPU gm --
    magma_malloc( (void**) &d_startSols_array,  batchCount * sizeof(magmaDoubleComplex*) );
    magma_malloc( (void**) &d_Track_array,      batchCount * sizeof(magmaDoubleComplex*) );
    magma_malloc( (void**) &d_startCoefs_array,              sizeof(magmaDoubleComplex*) );
    magma_malloc( (void**) &d_targetCoefs_array,             sizeof(magmaDoubleComplex*) );
    magma_malloc( (void**) &d_cgesvA_array,     batchCount * sizeof(magmaDoubleComplex*) );
    magma_malloc( (void**) &d_cgesvB_array,     batchCount * sizeof(magmaDoubleComplex*) );

    // -- random initialization for h_cgesvA and h_cgesvB (doesn't matter the value) --
    lapackf77_zlarnv( &ione, ISEED, &sizeA, h_cgesvA );
    lapackf77_zlarnv( &ione, ISEED, &sizeB, h_cgesvB );

    // -- transfer data from CPU memory to GPU memory --
    data_h2d_time = magma_sync_wtime( my_queue );
    magma_zsetmatrix( N+1, batchCount, h_startSols, (N+1), d_startSols, (N+1), my_queue );
    magma_zsetmatrix( N+1, batchCount, h_Track, (N+1), d_Track, (N+1), my_queue );
    magma_zsetmatrix( coefsCount, 1, h_startParams, coefsCount, d_startCoefs, ldd_coefs, my_queue );
    magma_zsetmatrix( coefsCount, 1, h_targetParams, coefsCount, d_targetCoefs, ldd_coefs, my_queue );
    magma_zsetmatrix( N, N*batchCount, h_cgesvA, lda, d_cgesvA, ldda, my_queue );
    magma_zsetmatrix( N, batchCount,   h_cgesvB, ldb, d_cgesvB, lddb, my_queue );

    // -- connect pointer to 2d arrays --
    magma_zset_pointer( d_startSols_array, d_startSols, (N+1), 0, 0, (N+1), batchCount, my_queue );
    magma_zset_pointer( d_Track_array, d_Track, (N+1), 0, 0, (N+1), batchCount, my_queue );
    magma_zset_pointer( d_startCoefs_array, d_startCoefs, ldd_coefs, 0, 0, ldd_coefs, 1, my_queue );
    magma_zset_pointer( d_targetCoefs_array, d_targetCoefs, ldd_coefs, 0, 0, ldd_coefs, 1, my_queue );
    magma_zset_pointer( d_cgesvA_array, d_cgesvA, ldda, 0, 0, ldda*N, batchCount, my_queue );
    magma_zset_pointer( d_cgesvB_array, d_cgesvB, lddb, 0, 0, ldda, batchCount, my_queue );

    data_h2d_time = magma_sync_wtime( my_queue ) - data_h2d_time;
    //std::cout<<"Host to device data transfer time: "<<data_h2d_time*1000<<std::endl;

    // ===================================================================
    // magma GPU cgesv batched solver for Homotopy Continuation
    // ===================================================================
    std::cout<<"GPU computing ..."<<std::endl;

    if (hc_problem == "six_lines_6x6") {
      std::cout<<"Solving six lines 6x6 problem ..."<<std::endl<<std::endl;
      gpu_time = kernel_HC_Solver_DP64_six_lines_6x6(N, batchCount, ldda, my_queue, d_startSols_array, d_Track_array, d_cgesvA_array, d_cgesvB_array, 
                                                d_Hx_idx, d_Ht_idx, d_phc_coeffs_Hx, d_phc_coeffs_Ht, pp->numOfCoeffsFromParams);
    }
    else if (hc_problem == "3views_4pts") {
      std::cout<<"Solving 3 Views with 4 Points Problem ..."<<std::endl<<std::endl;
      gpu_time = kernel_HC_Solver_DP64_3views_4pts(N, batchCount, ldda, my_queue, d_startSols_array, d_Track_array, d_cgesvA_array, d_cgesvB_array, 
                                              d_Hx_idx, d_Ht_idx, d_phc_coeffs_Hx, d_phc_coeffs_Ht, pp->numOfCoeffsFromParams);
    }

    // -- check returns from the kernel --
    data_d2h_time = magma_sync_wtime( my_queue );
    magma_zgetmatrix( (N+1), batchCount, d_Track, (N+1), h_track_sols, (N+1), my_queue );
    magma_zgetmatrix( N, batchCount, d_cgesvB, lddb, h_cgesvB_verify, ldb, my_queue );
    magma_zgetmatrix( N, N*batchCount, d_cgesvA, ldda, h_cgesvA_verify, lda, my_queue );
    data_d2h_time = magma_sync_wtime( my_queue ) - data_d2h_time;
    std::cout<<"results:"<<std::endl;

    /*int num_of_convergence = 0;
    for (int bs = 0; bs < batchCount; bs++) {
      track_sols_file << std::setprecision(10);
      int num_of_real_sols = 0;
      
      if (MAGMA_Z_REAL((h_cgesvB_verify + bs * ldb)[0]) == 1) {
        track_sols_file << MAGMA_Z_IMAG((h_cgesvB_verify + bs*N)[0]) << "\t" << std::setprecision(20) << bs << "\n";
        for (int vs = 0; vs < N; vs++) {
          track_sols_file << std::setprecision(20) << MAGMA_Z_REAL((h_track_sols + bs * (N+1))[vs]) << "\t" << std::setprecision(20) << MAGMA_Z_IMAG((h_track_sols + bs * (N+1))[vs]) << "\n";
        }
        track_sols_file << "\n";
        num_of_convergence++;

      }
    }   */

    
    int num_of_convergence = 0;
    for (int bs = 0; bs < batchCount; bs++) {

      int num_of_real_sols = 0;

      //> Make sure the HC path is converged
      if (MAGMA_Z_REAL((h_cgesvB_verify + bs * ldb)[0]) == 1) {

        //> Count the number of positive real variables in the solution
        for (int vs = 0; vs < N; vs++) {
          if ((MAGMA_Z_IMAG((h_track_sols + bs * (N+1))[vs]) < 0.01) && (MAGMA_Z_REAL((h_track_sols + bs * (N+1))[vs]) >= 0)) {
            num_of_real_sols++;
          }
        }
        
        //> If the number of real variables equals the number of variables, it is a real solution
        if (num_of_real_sols == N) {
          for (int vs = 0; vs < N; vs++) {
            track_sols_file << std::setprecision(20) << MAGMA_Z_REAL((h_track_sols + bs * (N+1))[vs]) << "\n";
          }
          track_sols_file << "\n";
        }
      }
      
    }
    

    std::cout<< "Number of convergence: " << num_of_convergence <<std::endl;
    printf("GPU time = %7.2f (ms)\n", (gpu_time)*1000);

    timings_file << std::setprecision(20) << (gpu_time)*1000 << "\n";

    magma_queue_destroy( my_queue );

    magma_free_cpu( h_cgesvA );
    magma_free_cpu( h_cgesvB );
    magma_free_cpu( h_cgesvA_verify );
    magma_free_cpu( h_cgesvB_verify );
    magma_free_cpu( h_track_sols );

    magma_free( d_startSols );
    magma_free( d_Track );
    magma_free( d_startCoefs );
    magma_free( d_targetCoefs );
    magma_free( d_cgesvA );
    magma_free( d_cgesvB );
    magma_free( d_cgesvA_array );
    magma_free( d_cgesvB_array );
    magma_free( d_Hx_idx );
    magma_free( d_Ht_idx );

    magma_free(d_phc_coeffs_Hx);
    magma_free(d_phc_coeffs_Ht);

    fflush( stdout );
    printf( "\n" );
    magma_finalize();
  }

} // end of namespace

#endif
