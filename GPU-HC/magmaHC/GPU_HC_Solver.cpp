#ifndef GPU_HC_Solver_cpp
#define GPU_HC_Solver_cpp
// =============================================================================================================================
//
// ChangLogs
//    22-10-18:   Initially Created (Copied from other repos)
//    23-12-28:   Use macros and organize this file as definitions of GPU_HC_Solver class functions
//    23-12-29:   Change the file name to GPU_HC_Solver.cpp as a pool of defining member functions in class GPU_HC_Solver.hpp
//    24-02-26:   Add Data Reader to clean up the main code.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =============================================================================================================================
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

//> MAGMA
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_internal.h"
#undef max
#undef min

#include "GPU_HC_Solver.hpp"
#include "definitions.hpp"
#include "gpu-kernels/magmaHC-kernels.hpp"

//> Constructor
GPU_HC_Solver::GPU_HC_Solver(YAML::Node Problem_Settings_File): Problem_Setting_YAML_File(Problem_Settings_File) {

    //> Parse data from the YAML file
    //> (1) Problem Name and GPU-HC Type
    HC_problem                      = Problem_Setting_YAML_File["problem_name"].as<std::string>();
    HC_print_problem_name           = Problem_Setting_YAML_File["problem_print_out_name"].as<std::string>();
    //> (2) GPU-HC Configurations
    GPUHC_Max_Steps                 = Problem_Setting_YAML_File["GPUHC_Max_Steps"].as<int>();
    GPUHC_Max_Correction_Steps      = Problem_Setting_YAML_File["GPUHC_Max_Correction_Steps"].as<int>();
    GPUHC_delta_t_incremental_steps = Problem_Setting_YAML_File["GPUHC_Num_Of_Steps_to_Increase_Delta_t"].as<int>();
    //> (3) Problem Specifications
    Num_Of_Vars                     = Problem_Setting_YAML_File["Num_Of_Vars"].as<int>();
    Num_Of_Params                   = Problem_Setting_YAML_File["Num_Of_Params"].as<int>();
    Num_Of_Tracks                   = Problem_Setting_YAML_File["Num_Of_Tracks"].as<int>();
    dHdx_Max_Terms                  = Problem_Setting_YAML_File["dHdx_Max_Terms"].as<int>();
    dHdx_Max_Parts                  = Problem_Setting_YAML_File["dHdx_Max_Parts"].as<int>();
    dHdt_Max_Terms                  = Problem_Setting_YAML_File["dHdt_Max_Terms"].as<int>();
    dHdt_Max_Parts                  = Problem_Setting_YAML_File["dHdt_Max_Parts"].as<int>();
    Max_Order_Of_T                  = Problem_Setting_YAML_File["Max_Order_Of_T"].as<int>();
    Num_Of_Coeffs_From_Params       = Problem_Setting_YAML_File["Num_Of_Coeffs_From_Params"].as<int>();
    //> (4) RANSAC data
    RANSAC_Dataset_Name             = Problem_Setting_YAML_File["RANSAC_Dataset"].as<std::string>();

    //> Initialization
    magma_init();
    magma_print_environment();

    magma_getdevice( &cdev );
    magma_queue_create( cdev, &my_queue );

    //> Define the array sizes
    dHdx_Index_Size       = Num_Of_Vars*Num_Of_Vars*dHdx_Max_Terms*dHdx_Max_Parts;
    dHdt_Index_Size       = Num_Of_Vars*dHdt_Max_Terms*dHdt_Max_Parts;
    dHdx_PHC_Coeffs_Size  = (Num_Of_Coeffs_From_Params+1)*(Max_Order_Of_T+1);
    dHdt_PHC_Coeffs_Size  = (Num_Of_Coeffs_From_Params+1)*(Max_Order_Of_T);
    ldd_phc_Params_Hx     = magma_roundup( dHdx_PHC_Coeffs_Size, 32 );  // multiple of 32 by default
    ldd_phc_Params_Ht     = magma_roundup( dHdt_PHC_Coeffs_Size, 32 );  // multiple of 32 by default

    printf("dHdx_Index_Size      = %5.2f KB\n", (double)(dHdx_Index_Size*sizeof(int)) / 1024.);
    printf("dHdt_Index_Size      = %5.2f KB\n", (double)(dHdt_Index_Size*sizeof(int)) / 1024.);
    printf("dHdx_PHC_Coeffs_Size = %5.2f KB\n", (double)(dHdx_PHC_Coeffs_Size*sizeof(magmaComplex)) / 1024.);
    printf("dHdt_PHC_Coeffs_Size = %5.2f KB\n", (double)(dHdt_PHC_Coeffs_Size*sizeof(magmaComplex)) / 1024.);

    magmaComplex **d_Start_Sols_array      = NULL;
    magmaComplex **d_Homotopy_Sols_array   = NULL;

    //> Define problem file path for problem data reader and output file path for results evaluations
    Problem_File_Path       = std::string("../../problems/") + HC_problem;
    RANSAC_Data_File_Path   = std::string("../../RANSAC_Data/") + HC_problem + "/" + RANSAC_Dataset_Name;
    Write_Files_Path        = std::string("../../") + WRITE_FILES_FOLDER;

    //> Evaluations
    Evaluate_GPUHC_Sols = std::shared_ptr<Evaluations>(new Evaluations(Write_Files_Path, Num_Of_Tracks, Num_Of_Vars));
}

void GPU_HC_Solver::Allocate_Arrays() {
    LOG_INFOR_MESG("Allocating arrays ...");
    //> CPU Allocations
#if USE_SINGLE_PRECISION
    magma_cmalloc_cpu( &h_Start_Sols,           Num_Of_Tracks*(Num_Of_Vars+1) );
    magma_cmalloc_cpu( &h_Homotopy_Sols,        Num_Of_Tracks*(Num_Of_Vars+1)*NUM_OF_RANSAC_ITERATIONS );
    magma_cmalloc_cpu( &h_Start_Params,         Num_Of_Params+1 );
    magma_cmalloc_cpu( &h_Target_Params,        (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS );
    magma_cmalloc_cpu( &h_dHdx_PHC_Coeffs,      dHdx_PHC_Coeffs_Size );
    magma_cmalloc_cpu( &h_dHdt_PHC_Coeffs,      dHdt_PHC_Coeffs_Size );
    magma_cmalloc_cpu( &h_diffParams,           (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS );
    magma_cmalloc_cpu( &h_GPU_HC_Track_Sols,    Num_Of_Tracks*(Num_Of_Vars+1)*NUM_OF_RANSAC_ITERATIONS ); //> Use to store GPU results from the CPU side
    magma_cmalloc_cpu( &h_Debug_Purpose,        Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS );
#else
    magma_zmalloc_cpu( &h_Start_Sols,           Num_Of_Tracks*(Num_Of_Vars+1) );
    magma_zmalloc_cpu( &h_Homotopy_Sols,        Num_Of_Tracks*(Num_Of_Vars+1)*NUM_OF_RANSAC_ITERATIONS );
    magma_zmalloc_cpu( &h_Start_Params,         Num_Of_Params+1 );
    magma_zmalloc_cpu( &h_Target_Params,        (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS );
    magma_zmalloc_cpu( &h_dHdx_PHC_Coeffs,      dHdx_PHC_Coeffs_Size );
    magma_zmalloc_cpu( &h_dHdt_PHC_Coeffs,      dHdt_PHC_Coeffs_Size );
    magma_zmalloc_cpu( &h_diffParams,           (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS );
    magma_zmalloc_cpu( &h_GPU_HC_Track_Sols,    Num_Of_Tracks*(Num_Of_Vars+1)*NUM_OF_RANSAC_ITERATIONS ); //> Use to store GPU results from the CPU side
    magma_zmalloc_cpu( &h_Debug_Purpose,        Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS );
#endif

    h_dHdx_Index              = new int[ dHdx_Index_Size ];
    h_dHdt_Index              = new int[ dHdt_Index_Size ];
    h_is_GPU_HC_Sol_Converge  = new bool[ Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS ];
    h_is_GPU_HC_Sol_Infinity  = new bool[ Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS ];

    //> GPU Allocations
#if USE_SINGLE_PRECISION
    magma_cmalloc( &d_Start_Sols,               Num_Of_Tracks*(Num_Of_Vars+1) );
    magma_cmalloc( &d_Homotopy_Sols,            Num_Of_Tracks*(Num_Of_Vars+1)*NUM_OF_RANSAC_ITERATIONS );
    magma_cmalloc( &d_Start_Params,             Num_Of_Params+1 );
    magma_cmalloc( &d_Target_Params,            (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS );
    magma_cmalloc( &d_dHdx_PHC_Coeffs,          ldd_phc_Params_Hx );
    magma_cmalloc( &d_dHdt_PHC_Coeffs,          ldd_phc_Params_Ht );
    magma_cmalloc( &d_diffParams,               (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS );
    magma_cmalloc( &d_Debug_Purpose,            Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS );
#else
    magma_zmalloc( &d_Start_Sols,               Num_Of_Tracks*(Num_Of_Vars+1) );
    magma_zmalloc( &d_Homotopy_Sols,            Num_Of_Tracks*(Num_Of_Vars+1)*NUM_OF_RANSAC_ITERATIONS );
    magma_zmalloc( &d_Start_Params,             Num_Of_Params+1 );
    magma_zmalloc( &d_Target_Params,            (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS );
    magma_zmalloc( &d_dHdx_PHC_Coeffs,          ldd_phc_Params_Hx );
    magma_zmalloc( &d_dHdt_PHC_Coeffs,          ldd_phc_Params_Ht );
    magma_zmalloc( &d_diffParams,               (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS );
    magma_zmalloc( &d_Debug_Purpose,            Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS );
#endif
    cudacheck( cudaMalloc( &d_dHdx_Index,       dHdx_Index_Size *sizeof(int)) );
    cudacheck( cudaMalloc( &d_dHdt_Index,       dHdt_Index_Size *sizeof(int)) );

    magma_malloc( (void**) &d_Start_Sols_array,     (Num_Of_Tracks)                              * sizeof(magmaComplex*) );
    magma_malloc( (void**) &d_Homotopy_Sols_array,  (Num_Of_Tracks)*(NUM_OF_RANSAC_ITERATIONS) * sizeof(magmaComplex*) );

    cudacheck( cudaMalloc( &d_is_GPU_HC_Sol_Converge, (Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS) * sizeof(bool) ));
    cudacheck( cudaMalloc( &d_is_GPU_HC_Sol_Infinity, (Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS) * sizeof(bool) ));
}

bool GPU_HC_Solver::Read_Problem_Data() {
    LOG_INFOR_MESG("Reading problem data ...");
    //> Load problem data to arrays
    bool is_Data_Read_Successfully = false;

    //> Data reader
    Load_Problem_Data = std::shared_ptr<Data_Reader>(new Data_Reader(Problem_File_Path, RANSAC_Data_File_Path, Num_Of_Tracks, Num_Of_Vars, Num_Of_Params));

    //> (1) Start parameters
    is_Data_Read_Successfully = Load_Problem_Data->Read_Start_Params( h_Start_Params );
    if (!is_Data_Read_Successfully) { LOG_DATA_LOAD_ERROR("Start Parameters"); return false; }

    //> (2) Target parameters, if required
    if (RANSAC_Dataset_Name == "None") {
        is_Data_Read_Successfully = Load_Problem_Data->Read_Target_Params( h_Target_Params );
        if (!is_Data_Read_Successfully) { LOG_DATA_LOAD_ERROR("Target Parameters"); return false; }
        for (int i = 0; i <= Num_Of_Params; i++) 
            h_diffParams[i] = h_Target_Params[i] - h_Start_Params[i];
    }

    //> (3) Start solutions
    is_Data_Read_Successfully = Load_Problem_Data->Read_Start_Sols( h_Start_Sols, h_Homotopy_Sols );
    if (!is_Data_Read_Successfully) { LOG_DATA_LOAD_ERROR("Start Solutions"); return false; }

    //> (4) dH/dx evaluation indices
    is_Data_Read_Successfully = Load_Problem_Data->Read_dHdx_Indices( h_dHdx_Index );
    if (!is_Data_Read_Successfully) { LOG_DATA_LOAD_ERROR("dH/dx Evaluation Indices"); return false; }

    //> (5) dH/dt evaluation indices
    is_Data_Read_Successfully = Load_Problem_Data->Read_dHdt_Indices( h_dHdt_Index );
    if (!is_Data_Read_Successfully) { LOG_DATA_LOAD_ERROR("dH/dt Evaluation Indices"); return false; }

    //> (6) Coefficients from parameters, if required
    if (!Load_Problem_Data->Construct_Coeffs_From_Params( HC_problem, h_Target_Params, h_Start_Params, h_dHdx_PHC_Coeffs, h_dHdt_PHC_Coeffs )) return false;

    return true;
}

void GPU_HC_Solver::Data_Transfer_From_Host_To_Device() {
    LOG_INFOR_MESG("Transfering data from host to device ...");
    transfer_h2d_time = magma_sync_wtime( my_queue );
#if USE_SINGLE_PRECISION
    magma_csetmatrix( Num_Of_Vars+1,   Num_Of_Tracks,                           h_Start_Sols,     (Num_Of_Vars+1),  d_Start_Sols,    Num_Of_Vars+1,     my_queue );
    magma_csetmatrix( Num_Of_Vars+1,   Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS,  h_Homotopy_Sols,  (Num_Of_Vars+1),  d_Homotopy_Sols, Num_Of_Vars+1,     my_queue );
    magma_csetmatrix( Num_Of_Params+1, (1),                                     h_Start_Params,   Num_Of_Params+1,  d_Start_Params,  Num_Of_Params+1,   my_queue );
    magma_csetmatrix( (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS, (1), h_diffParams,    (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS, d_diffParams,    (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS, my_queue );
    magma_csetmatrix( (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS, (1), h_Target_Params, (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS, d_Target_Params, (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS, my_queue );
    //> connect pointer to 2d arrays
    magma_cset_pointer( d_Start_Sols_array,    d_Start_Sols,     (Num_Of_Vars+1), 0, 0, (Num_Of_Vars+1), Num_Of_Tracks, my_queue );
    magma_cset_pointer( d_Homotopy_Sols_array, d_Homotopy_Sols,  (Num_Of_Vars+1), 0, 0, (Num_Of_Vars+1), Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS, my_queue );
    magma_csetmatrix( dHdx_PHC_Coeffs_Size, (1),       h_dHdx_PHC_Coeffs,   dHdx_PHC_Coeffs_Size, d_dHdx_PHC_Coeffs,  ldd_phc_Params_Hx, my_queue );
    magma_csetmatrix( dHdt_PHC_Coeffs_Size, (1),       h_dHdt_PHC_Coeffs,   dHdt_PHC_Coeffs_Size, d_dHdt_PHC_Coeffs,  ldd_phc_Params_Ht, my_queue );
#else
    magma_zsetmatrix( Num_Of_Vars+1,   Num_Of_Tracks,                           h_Start_Sols,     (Num_Of_Vars+1),  d_Start_Sols,    Num_Of_Vars+1,     my_queue );
    magma_zsetmatrix( Num_Of_Vars+1,   Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS,  h_Homotopy_Sols,  (Num_Of_Vars+1),  d_Homotopy_Sols, Num_Of_Vars+1,     my_queue );
    magma_zsetmatrix( Num_Of_Params+1, (1),                                     h_Start_Params,   Num_Of_Params+1,  d_Start_Params,  Num_Of_Params+1,   my_queue );
    magma_zsetmatrix( (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS, (1), h_diffParams,    (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS, d_diffParams,    (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS, my_queue );
    magma_zsetmatrix( (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS, (1), h_Target_Params, (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS, d_Target_Params, (Num_Of_Params+1)*NUM_OF_RANSAC_ITERATIONS, my_queue );
    //> connect pointer to 2d arrays
    magma_zset_pointer( d_Start_Sols_array,    d_Start_Sols,     (Num_Of_Vars+1), 0, 0, (Num_Of_Vars+1), Num_Of_Tracks, my_queue );
    magma_zset_pointer( d_Homotopy_Sols_array, d_Homotopy_Sols,  (Num_Of_Vars+1), 0, 0, (Num_Of_Vars+1), Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS, my_queue );
    magma_zsetmatrix( dHdx_PHC_Coeffs_Size, (1),       h_dHdx_PHC_Coeffs,   dHdx_PHC_Coeffs_Size, d_dHdx_PHC_Coeffs,  ldd_phc_Params_Hx, my_queue );
    magma_zsetmatrix( dHdt_PHC_Coeffs_Size, (1),       h_dHdt_PHC_Coeffs,   dHdt_PHC_Coeffs_Size, d_dHdt_PHC_Coeffs,  ldd_phc_Params_Ht, my_queue );
#endif
    
    cudacheck( cudaMemcpy( d_dHdx_Index, h_dHdx_Index,     dHdx_Index_Size * sizeof(int), cudaMemcpyHostToDevice) );
    cudacheck( cudaMemcpy( d_dHdt_Index, h_dHdt_Index,     dHdt_Index_Size * sizeof(int), cudaMemcpyHostToDevice) );
    transfer_h2d_time = magma_sync_wtime( my_queue ) - transfer_h2d_time;
}

void GPU_HC_Solver::Solve_by_GPU_HC() {
    LOG_INFOR_MESG("GPU computing ...");
    if (HC_problem == "generalized_3views_4pts") {
        gpu_time = kernel_HC_Solver_generalized_3views_4pts
                   (my_queue, GPUHC_Max_Steps, GPUHC_Max_Correction_Steps, GPUHC_delta_t_incremental_steps, \
                    d_Start_Sols_array, d_Homotopy_Sols_array, \
                    d_dHdx_Index, d_dHdt_Index, d_dHdx_PHC_Coeffs, d_dHdt_PHC_Coeffs, \
                    d_is_GPU_HC_Sol_Converge, d_is_GPU_HC_Sol_Infinity, d_Debug_Purpose);
    }
    else if (HC_problem == "generalized_3views_6lines") {
        gpu_time = kernel_HC_Solver_generalized_3views_6lines
                   (my_queue, GPUHC_Max_Steps, GPUHC_Max_Correction_Steps, GPUHC_delta_t_incremental_steps, \
                    d_Start_Sols_array, d_Homotopy_Sols_array, \
                    d_dHdx_Index, d_dHdt_Index, d_dHdx_PHC_Coeffs, d_dHdt_PHC_Coeffs, \
                    d_is_GPU_HC_Sol_Converge, d_is_GPU_HC_Sol_Infinity, d_Debug_Purpose);
    }

    //> Check returns from the GPU kernel
    transfer_d2h_time = magma_sync_wtime( my_queue );
#if USE_SINGLE_PRECISION
    magma_cgetmatrix( (Num_Of_Vars+1), Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS, d_Homotopy_Sols,  (Num_Of_Vars+1), h_GPU_HC_Track_Sols,    (Num_Of_Vars+1), my_queue );
#else
    magma_zgetmatrix( (Num_Of_Vars+1), Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS, d_Homotopy_Sols,  (Num_Of_Vars+1), h_GPU_HC_Track_Sols,    (Num_Of_Vars+1), my_queue );
#endif
    cudacheck( cudaMemcpy( h_is_GPU_HC_Sol_Converge, d_is_GPU_HC_Sol_Converge, Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS*sizeof(bool), cudaMemcpyDeviceToHost) );
    cudacheck( cudaMemcpy( h_is_GPU_HC_Sol_Infinity, d_is_GPU_HC_Sol_Infinity, Num_Of_Tracks*NUM_OF_RANSAC_ITERATIONS*sizeof(bool), cudaMemcpyDeviceToHost) );
    transfer_d2h_time = magma_sync_wtime( my_queue ) - transfer_d2h_time;
    std::cout << "GPU-HC finishes." << std::endl;

#if GPU_DEBUG
#if USE_SINGLE_PRECISION
    magma_cgetmatrix( Num_Of_Tracks, NUM_OF_RANSAC_ITERATIONS, d_Debug_Purpose, Num_Of_Tracks, h_Debug_Purpose, Num_Of_Tracks, my_queue );
#else
    magma_zgetmatrix( Num_Of_Tracks, NUM_OF_RANSAC_ITERATIONS, d_Debug_Purpose, Num_Of_Tracks, h_Debug_Purpose, Num_Of_Tracks, my_queue );
#endif
#endif

    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "## Solving " << HC_print_problem_name << std::endl;

    //> Print out timings
    printf("## Timings:\n");
    printf(" - GPU Computation Time = %7.2f (ms)\n", (gpu_time)*1000);
    printf(" - H2D Transfer Time    = %7.2f (ms)\n", (transfer_h2d_time)*1000);
    printf(" - D2H Transfer Time    = %7.2f (ms)\n", (transfer_d2h_time)*1000);

    //> Object for the Evaluations class
    Evaluate_GPUHC_Sols->Write_Converged_Sols( h_GPU_HC_Track_Sols, h_is_GPU_HC_Sol_Converge );
    Evaluate_GPUHC_Sols->Flush_Out_Data();
    Evaluate_GPUHC_Sols->Evaluate_GPUHC_Sols( h_GPU_HC_Track_Sols, h_is_GPU_HC_Sol_Converge, h_is_GPU_HC_Sol_Infinity, 0 );
    Evaluate_GPUHC_Sols->Find_Unique_Sols( h_GPU_HC_Track_Sols, h_is_GPU_HC_Sol_Converge );

    //> Print out evaluation results
    std::cout << "## Evaluation of GPU-HC Solutions: "      << std::endl;
    std::cout << " - Number of Converged Solutions:       " << Evaluate_GPUHC_Sols->Num_Of_Coverged_Sols << std::endl;
    std::cout << " - Number of Real Solutions:            " << Evaluate_GPUHC_Sols->Num_Of_Real_Sols << std::endl;
    std::cout << " - Number of Infinity Failed Solutions: " << Evaluate_GPUHC_Sols->Num_Of_Inf_Sols << std::endl;
    std::cout << " - Number of Unique Solutions:          " << Evaluate_GPUHC_Sols->Num_Of_Unique_Sols << std::endl;
}

GPU_HC_Solver::~GPU_HC_Solver() {

    magma_queue_destroy( my_queue );

    delete [] h_is_GPU_HC_Sol_Converge;
    delete [] h_is_GPU_HC_Sol_Infinity;
    delete [] h_dHdx_Index;
    delete [] h_dHdt_Index;

    magma_free_cpu( h_Start_Sols );
    magma_free_cpu( h_Homotopy_Sols );
    magma_free_cpu( h_Start_Params );
    magma_free_cpu( h_Target_Params );
    magma_free_cpu( h_dHdx_PHC_Coeffs );
    magma_free_cpu( h_dHdt_PHC_Coeffs );
    magma_free_cpu( h_GPU_HC_Track_Sols );
    magma_free_cpu( h_Debug_Purpose );
    magma_free_cpu( h_diffParams );

    magma_free( d_diffParams );
    magma_free( d_is_GPU_HC_Sol_Converge );
    magma_free( d_is_GPU_HC_Sol_Infinity );
    magma_free( d_Start_Sols );
    magma_free( d_Homotopy_Sols );
    magma_free( d_Start_Params );
    magma_free( d_Target_Params );
    magma_free( d_dHdx_PHC_Coeffs );
    magma_free( d_dHdt_PHC_Coeffs );
    magma_free( d_Debug_Purpose );

    cudacheck( cudaFree( d_dHdx_Index ) );
    cudacheck( cudaFree( d_dHdt_Index ) );

    fflush( stdout );
    printf( "\n" );
    magma_finalize();
}

#endif
