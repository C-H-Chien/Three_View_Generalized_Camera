#ifndef DATA_READER_CPP
#define DATA_READER_CPP
// ============================================================================
// Data_Reader class CPP: read data from the problem files
//
// Changelogs
//    Chien  24-01-21:   Initially Created.
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
#include <cuComplex.h>

#include "magma_v2.h"
#include "Data_Reader.hpp"
#include "./PHC_Coeffs/p2c-generalized_3views_4pts.h"
#include "./PHC_Coeffs/p2c-numerical_generalized_3views_6lines.h"

Data_Reader::Data_Reader(std::string Problem_Filename, std::string RANSAC_Data_File_Path, const int Num_Of_Tracks, const int Num_Of_Vars, const int Num_Of_Params) \
    : num_of_tracks(Num_Of_Tracks), num_of_variables(Num_Of_Vars), num_of_params(Num_Of_Params), RANSAC_Data_Path_(RANSAC_Data_File_Path) {

    //> Define problem file names
    File_Name_Start_Params = Problem_Filename + std::string("/start_params.txt");
    File_Name_Target_Params = Problem_Filename + std::string("/target_params.txt");
    File_Name_Start_Sols = Problem_Filename + std::string("/start_sols.txt");

    File_Name_dHdx_Indx = Problem_Filename + std::string("/dHdx_indx.txt");
    File_Name_dHdt_Indx = Problem_Filename + std::string("/dHdt_indx.txt");

    File_Name_Intrinsic_Matrix = RANSAC_Data_Path_ + std::string("/Intrinsic_Matrix.txt");
}

bool Data_Reader::Construct_Coeffs_From_Params( std::string HC_Problem, \
        magmaComplex* h_Target_Params,     magmaComplex* h_Start_Params, \
        magmaComplex* &h_dHdx_PHC_Coeffs,  magmaComplex* &h_dHdt_PHC_Coeffs ) 
{
    if (HC_Problem == "generalized_3views_4pts")           magmaHCWrapper::p2c_generalized_3views_4pts(h_Target_Params, h_Start_Params, h_dHdx_PHC_Coeffs, h_dHdt_PHC_Coeffs);
    else if (HC_Problem == "generalized_3views_6lines")    magmaHCWrapper::p2c_numerical_generalized_3views_6lines(h_dHdx_PHC_Coeffs, h_dHdt_PHC_Coeffs);
    else {
        LOG_ERROR("Invalid HC problem name or P2C function is not included in the Construct_Coeffs_From_Params function.");
        return false;
    }
    return true;
}

bool Data_Reader::Read_Start_Sols(magmaComplex* &h_Start_Sols, magmaComplex* &h_Homotopy_Sols) {
    //> Read start solutions
    File_Start_Sols.open(File_Name_Start_Sols, std::ios_base::in);
    if (!File_Start_Sols) {
        LOG_FILE_ERROR(File_Name_Start_Sols);
        return false;
    }
    else {
        FP_type s_real, s_imag;
        int d = 0, i = 0; 
        while (File_Start_Sols >> s_real >> s_imag) {
            (h_Start_Sols + i * (num_of_variables+1))[d]     = MAGMA_MAKE_COMPLEX(s_real, s_imag);
            (h_Homotopy_Sols + i * (num_of_variables+1))[d]  = MAGMA_MAKE_COMPLEX(s_real, s_imag);
            if (d < num_of_variables-1) d++;
            else {
                d = 0;
                i++;
            }
        }
        for(int k = 0; k < num_of_tracks; k++) {
            (h_Start_Sols + k * (num_of_variables+1))[num_of_variables]    = MAGMA_MAKE_COMPLEX(1.0, 0.0);
            (h_Homotopy_Sols + k * (num_of_variables+1))[num_of_variables] = MAGMA_MAKE_COMPLEX(1.0, 0.0);
        }
        
    }

    //> Copy the start solutions a number of RANSAC iterations times for h_Homotopy_Sols array
    for (int ri = 1; ri < NUM_OF_RANSAC_ITERATIONS; ri++) {
        for (int i = 0; i < (num_of_tracks * (num_of_variables + 1)); i++) {
            (h_Homotopy_Sols + ri * (num_of_tracks * (num_of_variables + 1)))[i] = h_Homotopy_Sols[i];
        }
    }
#if RANSAC_DEBUG
    int copy_id = 2;
    if (NUM_OF_RANSAC_ITERATIONS > copy_id) {
        std::cout << "Printing coopied h_Homotopy_Sols:" << std::endl;
        printf("     Copy 1                       Copy 2\n");
        for (int i = 0; i < num_of_variables+1; i++) {
            printf(" (%.7f, %.7f)        (%.7f, %.7f)\n", MAGMA_COMPLEX_REAL(h_Homotopy_Sols[i]), MAGMA_COMPLEX_IMAG(h_Homotopy_Sols[i]), \
                                                        MAGMA_COMPLEX_REAL((h_Homotopy_Sols + copy_id*(num_of_tracks * (num_of_variables + 1)))[i]), \
                                                        MAGMA_COMPLEX_IMAG((h_Homotopy_Sols + copy_id*(num_of_tracks * (num_of_variables + 1)))[i]));
        }
    }
#endif
    return true;
}

bool Data_Reader::Read_Target_Params(magmaComplex* &h_Target_Params) {
    int d = 0;
    File_Target_Params.open(File_Name_Target_Params, std::ios_base::in);
    if (!File_Target_Params) {
        LOG_FILE_ERROR(File_Name_Target_Params);
        return false;
    }
    else {
        FP_type s_real, s_imag;
        while (File_Target_Params >> s_real >> s_imag) {
            h_Target_Params[d] = MAGMA_MAKE_COMPLEX(s_real, s_imag);
            d++;
        }
        h_Target_Params[num_of_params] = MAGMA_COMPLEX_ONE;
    }

    // std::cout << "Target parameters:" << std::endl;
    // for (int i = 0; i <= num_of_params; i++)
    //     std::cout << MAGMA_COMPLEX_REAL(h_Target_Params[i]) << "\t" << MAGMA_COMPLEX_IMAG(h_Target_Params[i]) << std::endl;
    // std::cout << std::endl;
    return true;
}

bool Data_Reader::Read_Start_Params(magmaComplex* &h_Start_Params) {
    int d = 0;
    File_Start_Params.open(File_Name_Start_Params, std::ios_base::in);
    // LOG_INFOR_MESG("Start params file name: " + File_Name_Start_Params);
    if (!File_Start_Params) {
        LOG_FILE_ERROR(File_Name_Start_Params);
        return false;
    }
    else {
        FP_type s_real, s_imag;
        while (File_Start_Params >> s_real >> s_imag) {
            h_Start_Params[d] = MAGMA_MAKE_COMPLEX(s_real, s_imag);
            d++;
        }
        h_Start_Params[num_of_params] = MAGMA_COMPLEX_ONE;
    }

    // std::cout << "Start parameters:" << std::endl;
    // for (int i = 0; i <= num_of_params; i++)
    //     std::cout << MAGMA_COMPLEX_REAL(h_Start_Params[i]) << "\t" << MAGMA_COMPLEX_IMAG(h_Start_Params[i]) << std::endl;
    // std::cout << std::endl;
    return true;
}

bool Data_Reader::Read_dHdx_Indices( int* &h_dHdx_Index ) {
    int index, d = 0;
    File_dHdx_Indices.open(File_Name_dHdx_Indx, std::ios_base::in);
    if (!File_dHdx_Indices) {
        LOG_FILE_ERROR(File_Name_dHdx_Indx);
        return false;
    }
    else {
        while (File_dHdx_Indices >> index) {
            (h_dHdx_Index)[d] = (int)index;
            d++;
        }
#if DATA_READER_DEBUG
    std::cout << "Printing h_dHdx_Index ..." << std::endl;
    for (int i = 0; i < 10; i++) printf("%d\t", (int)h_dHdx_Index[i]);
    std::cout << std::endl;
#endif
        return true;
    }
}

bool Data_Reader::Read_dHdt_Indices( int* &h_dHdt_Index ) {
    int index, d = 0;
    File_dHdt_Indices.open(File_Name_dHdt_Indx, std::ios_base::in);
    if (!File_dHdt_Indices) {
        LOG_FILE_ERROR(File_Name_dHdt_Indx);
        return false;
    }
    else {
        while (File_dHdt_Indices >> index) {
            (h_dHdt_Index)[d] = (int)index;
            d++;
        }
#if DATA_READER_DEBUG
    std::cout << "Printing h_dHdt_Index ..." << std::endl;
    for (int i = 0; i < 10; i++) printf("%d\t", (int)h_dHdt_Index[i]);
    std::cout << std::endl;
#endif
        return true;
    }
}

// bool Data_Reader::Read_Camera_Matrices( float Pose21[12], float Pose31[12], float K[9], int tp_index ) {

//     //> Create padded file index
//     std::string str_File_Index = std::to_string(tp_index);
//     int min_str_length = (3 < str_File_Index.length()) ? 3 : str_File_Index.length();
//     auto padded_Index = std::string(3 - min_str_length, '0') + str_File_Index;
//     File_Name_Pose21  = RANSAC_Data_Path_ + "/GT_Poses21/GT_Poses21_" + padded_Index + ".txt";
//     File_Name_Pose31  = RANSAC_Data_Path_ + "/GT_Poses31/GT_Poses31_" + padded_Index + ".txt";

//     //> Intrinsic matrix
//     File_Intrinsic_Matrix.open(File_Name_Intrinsic_Matrix, std::ios_base::in);
//     if (!File_Intrinsic_Matrix) {
//         LOG_FILE_ERROR(File_Name_Intrinsic_Matrix);
//         return false;
//     }
//     else {
//         int d = 0;
//         float entry = 0.0;
//         while (File_Intrinsic_Matrix >> entry) {
//             K[d] = entry;
//             d++;
//         }
//     }

//     //> Extrinsic matrix - Camera relative pose view 1 & 2
//     File_Pose21.open(File_Name_Pose21, std::ios_base::in);
//     if (!File_Pose21) {
//         LOG_FILE_ERROR(File_Name_Pose21);
//         return false;
//     }
//     else {
//         int d = 0;
//         float entry = 0.0;
//         while (File_Pose21 >> entry) {
//             Pose21[d] = entry;
//             d++;
//         }
//     }

//     //> Extrinsic matrix - Camera relative pose view 1 & 3
//     File_Pose31.open(File_Name_Pose31, std::ios_base::in);
//     if (!File_Pose31) {
//         LOG_FILE_ERROR(File_Name_Pose31);
//         return false;
//     }
//     else {
//         int d = 0;
//         float entry = 0.0;
//         while (File_Pose31 >> entry) {
//             Pose31[d] = entry;
//             d++;
//         }
//     }
// #if DATA_READER_DEBUG
//     std::cout << std::endl << "Printing K ..." << std::endl;
//     for (int i = 0; i < 3; i++) {
//         for (int j = 0; j < 3; j++)
//             printf("%.6f\t", K[i*3 + j]);
//         printf("\n");
//     }
//     std::cout << std::endl;
//     std::cout << "Printing Pose21 ..." << std::endl;
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 3; j++)
//             printf("%.6f\t", Pose21[i*3 + j]);
//         printf("\n");
//     }
//     std::cout << std::endl;
//     std::cout << "Printing Pose31 ..." << std::endl;
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 3; j++)
//             printf("%.6f\t", Pose31[i*3 + j]);
//         printf("\n");
//     }
// #endif
//     return true;
// }

#endif
