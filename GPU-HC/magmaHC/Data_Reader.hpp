#ifndef DATA_READER_H
#define DATA_READER_H
// ============================================================================
// Data_Reader class: read data from the problem files
//
// Changelogs
//    Chien  24-01-21:   Initially Created.
//    Chien  24-05-24:   Add coefficients construction from parameters.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <vector>
#include <tuple>

#include "definitions.hpp"
#include "typenames.hpp"

class Data_Reader {

public:
    //> Constructor
    Data_Reader(std::string, std::string, const int, const int, const int);

    bool Read_Start_Params( magmaComplex* &h_Start_Params );
    bool Read_Target_Params( magmaComplex* &h_Target_Params );
    bool Read_Start_Sols( magmaComplex* &h_Start_Sols, magmaComplex* &h_Homotopy_Sols );

    bool Read_dHdx_Indices( int* &h_dHdx_Index );
    bool Read_dHdt_Indices( int* &h_dHdt_Index );

    //> TODO: RANSAC Data
    // bool Read_Camera_Matrices( float Pose21[12], float Pose31[12], float K[9], int tp_index );

    bool Construct_Coeffs_From_Params( std::string HC_Problem, \
        magmaComplex* h_Target_Params,     magmaComplex* h_Start_Params, \
        magmaComplex* &h_dHdx_PHC_Coeffs,  magmaComplex* &h_dHdt_PHC_Coeffs );

private:
    //> File names
    std::string File_Name_Target_Params;
    std::string File_Name_Start_Params;
    std::string File_Name_Start_Sols;
    std::string File_Name_dHdx_Indx;
    std::string File_Name_dHdt_Indx;
    std::string File_Name_Intrinsic_Matrix;
    std::string File_Name_Pose21;
    std::string File_Name_Pose31;
    std::string File_Name_Triplet_Edgels;

    //> input streams from problem files
    std::fstream File_Start_Params;
    std::fstream File_Target_Params;
    std::fstream File_Start_Sols;
    std::fstream File_dHdx_Indices;
    std::fstream File_dHdt_Indices;
    std::fstream File_Intrinsic_Matrix;
    std::fstream File_Pose21;
    std::fstream File_Pose31;
    std::fstream File_Triplet_Edgels;

    const int num_of_tracks;
    const int num_of_variables;
    const int num_of_params;

    std::string RANSAC_Data_Path_;
};

#endif
