#ifndef GPU_HC_SOLVER_HPP
#define GPU_HC_SOLVER_HPP
// ============================================================================
// GPU-HC solver class definition
//
// Modifications
//    Chiang-Heng Chien  24-05-23:      Shifted from icl branch
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
#include <memory>

#include "magma_v2.h"
#include "gpu-kernels/magmaHC-kernels.hpp"
#include "typenames.hpp"
#include "Data_Reader.hpp"
#include "Evaluations.hpp"
#include <yaml-cpp/yaml.h>

class Data_Reader;
class Evaluations;

// template<typename magma_T>
// struct MAGMA_TypeName {
//     typedef typename std::conditional< std::is_same<magma_T, float>::value, magmaFloatComplex, magmaDoubleComplex>::type magmaComplex;
//     typedef typename std::conditional< std::is_same<magma_T, float>::value, magmaFloatComplex_ptr, magmaDoubleComplex_ptr>::type magmaComplex_ptr;
// };

class GPU_HC_Solver {
    
    magma_device_t cdev;       // variable to indicate current gpu id
    magma_queue_t my_queue;    // magma queue variable, internally holds a cuda stream and a cublas handle

    //> Varaibles as sizes of arrays
    magma_int_t             dHdx_Index_Size;
    magma_int_t             dHdt_Index_Size;
    magma_int_t             dHdx_PHC_Coeffs_Size;
    magma_int_t             dHdt_PHC_Coeffs_Size;
    magma_int_t             ldd_phc_Params_Hx;
    magma_int_t             ldd_phc_Params_Ht;

    //> Variables and arrays on the CPU side
    magmaComplex            *h_GPU_HC_Track_Sols;
    magmaComplex            *h_diffParams;
    magmaComplex            *h_Start_Sols;
    magmaComplex            *h_Homotopy_Sols;
    magmaComplex            *h_Start_Params;
    magmaComplex            *h_Target_Params;
    magmaComplex            *h_dHdx_PHC_Coeffs;
    magmaComplex            *h_dHdt_PHC_Coeffs;
    int                     *h_dHdx_Index;
    int                     *h_dHdt_Index;
    magmaComplex            *h_Debug_Purpose;
    bool                    *h_is_GPU_HC_Sol_Converge;
    bool                    *h_is_GPU_HC_Sol_Infinity;
    FP_type                  h_Camera_Intrinsic_Matrix[9];
    FP_type                  h_Camera_Pose21[12];
    FP_type                  h_Camera_Pose31[12];

    //> Variables and arrays on the GPU side
    magmaComplex_ptr        d_Start_Sols, d_Homotopy_Sols;
    magmaComplex_ptr        d_Start_Params, d_Target_Params;
    magmaComplex_ptr        d_dHdx_PHC_Coeffs;
    magmaComplex_ptr        d_dHdt_PHC_Coeffs;
    int                     *d_dHdx_Index;
    int                     *d_dHdt_Index;
    magmaComplex            **d_Start_Sols_array;
    magmaComplex            **d_Homotopy_Sols_array;
    magmaComplex            *d_diffParams;
    magmaComplex            *d_Debug_Purpose;
    bool                    *d_is_GPU_HC_Sol_Converge;
    bool                    *d_is_GPU_HC_Sol_Infinity;

public:

    //> The timers
    real_Double_t           gpu_time;
    real_Double_t           transfer_h2d_time;
    real_Double_t           transfer_d2h_time;

    //> Constructor
    GPU_HC_Solver( YAML::Node );
    
    //> Member functions
    bool Read_Problem_Data();
    // bool Read_RANSAC_Data( int tp_index );
    void Allocate_Arrays();
    // void Prepare_Target_Params( );
    void Data_Transfer_From_Host_To_Device();
    void Solve_by_GPU_HC();

    std::string RANSAC_Dataset_Name;

    //> Destructor
    ~GPU_HC_Solver();

private:
    std::shared_ptr<Data_Reader> Load_Problem_Data = nullptr;
    std::shared_ptr<Evaluations> Evaluate_GPUHC_Sols = nullptr;
    YAML::Node Problem_Setting_YAML_File;
    
    std::string Problem_File_Path;
    std::string RANSAC_Data_File_Path;
    std::string Write_Files_Path;

    std::string HC_problem;
    std::string HC_print_problem_name;
    std::string GPUHC_type;
    int GPUHC_Max_Steps;
    int GPUHC_Max_Correction_Steps;
    int GPUHC_delta_t_incremental_steps;
    int Num_Of_Vars;
    int Num_Of_Params;
    int Num_Of_Tracks;
    int dHdx_Max_Terms;
    int dHdx_Max_Parts;
    int dHdt_Max_Terms;
    int dHdt_Max_Parts;
    int Max_Order_Of_T;

    //> RANSAC data
    int Num_Of_Coeffs_From_Params;
    std::vector<int> GPUHC_Actual_Sols_Steps_Collections;
};

#endif
