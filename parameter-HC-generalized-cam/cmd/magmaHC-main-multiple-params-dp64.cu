#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
// =======================================================================================================
// main function
//
// Modifications
//    Chien  23-02-26    Read Multiple Target Parameter Files and Solve (i) 3-Views With 
//                       4-Points Problem and (ii) Six Lines 6x6 Problem.
//    Chien  24-03-02    For 3-views 4-points problem, only 583 solutions are needed to find 
//                       the actual solution of the problem. This is the same as stated in the paper.
//                       The change is to remove reading 3072 solutions and read 583 solutions.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================================================
//> nvidia cuda
#include <cuda.h>
#include <cuda_runtime.h>

//> magma
#include "magma_v2.h"

//> magma
#include "magmaHC-DP64/magmaHC-problems.cuh"

//> p2c
#include "magmaHC-DP64/const-matrices/p2c-symbolic_3views_4pts.h"

//> global repo directory
std::string repo_dir = "/users/cchien3/data/cchien3/bitbucket-repos/gpu-phc/parameter-HC-generalized-cam/";

int main(int argc, char **argv) {
 argc; ++argv;
  std::string arg;
  int argIndx = 0;
  int argTotal = 4;
  std::string HC_problem = "default";

  if (argc) {
    arg = std::string(*argv);
    if (arg == "-h" || arg == "--help") {
      magmaHCWrapperDP64::print_usage();
      exit(1);
    }
    else if (argc <= argTotal) {
      while(argIndx <= argTotal-1) {
        if (arg == "-p" || arg == "--problem") {
          argv++;
          arg = std::string(*argv);
          HC_problem = arg;
          argIndx+=2;
          break;
        }
        else {
          std::cerr<<"invalid input arguments! See examples: \n";
          magmaHCWrapperDP64::print_usage();
          exit(1);
        }
        argv++;
      }
    }
    else if (argc > argTotal) {
      std::cerr<<"too many arguments!\n";
      magmaHCWrapperDP64::print_usage();
      exit(1);
    }
  }
  else { magmaHCWrapperDP64::print_usage(); exit(1); }

  magmaDoubleComplex *h_startSols;
  magmaDoubleComplex *h_Track;
  magmaDoubleComplex *h_startParams;
  magmaDoubleComplex *h_targetParams;
  magmaDoubleComplex *h_phc_coeffs_Hx;
  magmaDoubleComplex *h_phc_coeffs_Ht;
  magma_int_t *h_Hx_idx;
  magma_int_t *h_Ht_idx;

  //> files to be read
  std::string repo_root_dir = repo_dir;
  repo_dir.append("problems/");
  std::string problem_filename = repo_dir.append(HC_problem);

  //> declare class objects (put the long lasting object in dynamic memory)
  magmaHCWrapperDP64::problem_params* pp = new magmaHCWrapperDP64::problem_params;
  magmaHCWrapperDP64::const_mats* cm = new magmaHCWrapperDP64::const_mats;

  pp->define_problem_params(problem_filename, HC_problem);

  //> allocate tracks and coeffs arrays in cpu
  magma_zmalloc_cpu( &h_startSols, pp->numOfTracks*(pp->numOfVars+1) );
  magma_zmalloc_cpu( &h_Track, pp->numOfTracks*(pp->numOfVars+1) );
  magma_zmalloc_cpu( &h_startParams, pp->numOfParams );
  magma_zmalloc_cpu( &h_targetParams, pp->numOfParams );

  magma_zmalloc_cpu( &h_phc_coeffs_Hx, (pp->numOfCoeffsFromParams+1)*(pp->max_orderOf_t+1) );
  magma_zmalloc_cpu( &h_phc_coeffs_Ht, (pp->numOfCoeffsFromParams+1)*(pp->max_orderOf_t) );
  magma_imalloc_cpu( &h_Hx_idx, pp->numOfVars*pp->numOfVars*pp->Hx_maximal_terms*pp->Hx_maximal_parts );
  magma_imalloc_cpu( &h_Ht_idx, pp->numOfVars*pp->Ht_maximal_terms*pp->Ht_maximal_parts );

  std::string startParam_fileName = problem_filename;
  std::string startSols_filename_test = problem_filename;
  if (HC_problem == "3views_4pts")        { 
    startSols_filename_test.append("/start_sols.txt");
    startParam_fileName.append("/start_params.txt");
  }
  else if (HC_problem == "six_lines_6x6") { 
    startSols_filename_test.append("/start_sols.txt");
    startParam_fileName.append("/start_params.txt");
  }
  
  
  std::fstream startCoef_file;
  std::fstream startSols_file;
  bool read_success = 0;
  bool start_sols_read_success = 0;
  bool start_coeffs_read_success = 0;
  
  double s_real, s_imag;
  int d = 0, i = 0; 
  startSols_file.open(startSols_filename_test, std::ios_base::in);
  if (!startSols_file) { std::cerr << "Problem start solutions file NOT existed!\n"; exit(1); }
  else {
    while (startSols_file >> s_real >> s_imag) {
      (h_startSols + i * (pp->numOfVars+1))[d] = MAGMA_Z_MAKE(s_real, s_imag);
      (h_Track + i * (pp->numOfVars+1))[d] = MAGMA_Z_MAKE(s_real, s_imag);
      if (d < pp->numOfVars-1) { d++; }
      else {
        d = 0;
        i++;
      }
    }
    for(int k = 0; k < pp->numOfTracks; k++) {
      (h_startSols + k * (pp->numOfVars+1))[pp->numOfVars] = MAGMA_Z_MAKE(1.0, 0.0);
      (h_Track + k * (pp->numOfVars+1))[pp->numOfVars] = MAGMA_Z_MAKE(1.0, 0.0);
    }
    start_sols_read_success = 1;
  }

  //> read start system parameters
  d = 0;
  startCoef_file.open(startParam_fileName, std::ios_base::in);
  if (!startCoef_file) { std::cerr << "Problem start parameters file NOT existed!\n"; exit(1); }
  else {
    while (startCoef_file >> s_real >> s_imag) {
      (h_startParams)[d] = MAGMA_Z_MAKE(s_real, s_imag);
      d++;
    }
    start_coeffs_read_success = 1;
  }

  //>-------------------------------------------------------------------------------------------------
  bool Hx_file_read_success = false;
  bool Ht_file_read_success = false;
  
  std::string filename_Hx = problem_filename;
  std::string filename_Ht = problem_filename;
  filename_Hx.append("/Hx_idx.txt");
  filename_Ht.append("/Ht_idx.txt");
  std::fstream Hx_idx_file;
  std::fstream Ht_idx_file;
  
  //> 4) read Hx index matrix, if required
  int index;
  d = 0;
  Hx_idx_file.open(filename_Hx, std::ios_base::in);
  if (!Hx_idx_file) { std::cerr << "problem Hx index matrix file not existed!\n"; exit(1); }
  else {
    while (Hx_idx_file >> index) {
      (h_Hx_idx)[d] = index;
      d++;
    }
    Hx_file_read_success = 1;
  }
  //> 5) read Ht index matrix
  d = 0;
  Ht_idx_file.open(filename_Ht, std::ios_base::in);
  if (!Ht_idx_file) { std::cerr << "problem Ht index matrix file not existed!\n"; exit(1);}
  else {
    while (Ht_idx_file >> index) {
      (h_Ht_idx)[d] = index;
      d++;
    }
    Ht_file_read_success = 1;
  }

  //> Write the timings to a file
  std::ofstream timings_file;
  std::string write_timings_file_dir = repo_root_dir;
  write_timings_file_dir.append("Timings_Collection.txt");
  timings_file.open(write_timings_file_dir);
  if ( !timings_file.is_open() ) { std::cout<<"Collection of Timings write files cannot be opened!"<<std::endl; exit(1); }
  
  // =============================================================================
  //> read file: target parameters. Read them iteractively
  // =============================================================================
  bool RANSAC = true;
  int ratio = 90;
  int numOfTargetFiles = 250;
  problem_filename.append("/synthetic_evaluation/");
  for (int tp = 0; tp < numOfTargetFiles; tp++) {

    std::string targetParam_fileName;
    if (RANSAC) { 
      targetParam_fileName = std::string("/users/cchien3/data/cchien3/bitbucket-repos/gpu-phc/auto-gen-tools/RANSAC/outlier_ratio_");
      targetParam_fileName.append(std::to_string(ratio));
      targetParam_fileName.append("/target_params/target_params_"); 
    }
    else { targetParam_fileName = problem_filename; targetParam_fileName.append("/target_params_synthetic/target_params_"); }
    
    //targetParam_fileName.append("target_params_synthetic/target_params_");
    std::string fileIndx = std::to_string(tp);
    std::string extension = ".txt";
    std::string padded_fileName = std::string(6 - std::min(6, (int)(fileIndx.length())), '0') + fileIndx + extension;
    targetParam_fileName.append(padded_fileName);
    std::fstream targetParams_file;
    bool targetParams_read_success = false;

    d = 0;
    targetParams_file.open(targetParam_fileName, std::ios_base::in);
    if (!targetParams_file) { std::cerr << "problem target parameters file not existed!\n"; std::cout << targetParam_fileName << std::endl; exit(1); }
    else {
      while (targetParams_file >> s_real >> s_imag) {
        (h_targetParams)[d] = MAGMA_Z_MAKE(s_real, s_imag);
        d++;
      }
      targetParams_read_success = true;
    }

    //> PHC
    if (HC_problem == "3views_4pts")        { magmaHCWrapperDP64::p2c_symbolic_3views_4pts(h_startParams, h_targetParams, h_phc_coeffs_Hx, h_phc_coeffs_Ht); }
    else if (HC_problem == "six_lines_6x6") { 
      std::fstream phc_coeffs_Hx_file;
      std::fstream phc_coeffs_Ht_file;
      std::string phcHxCoeffs_fileName = problem_filename;
      std::string phcHtCoeffs_fileName = problem_filename;
      phcHxCoeffs_fileName.append("/Hx_numerical_data/Hx_");
      phcHtCoeffs_fileName.append("/Ht_numerical_data/Ht_");
      phcHxCoeffs_fileName.append(padded_fileName);
      phcHtCoeffs_fileName.append(padded_fileName);
      phc_coeffs_Hx_file.open(phcHxCoeffs_fileName, std::ios_base::in);
      phc_coeffs_Ht_file.open(phcHtCoeffs_fileName, std::ios_base::in);
      d = 0;
      //> Hx
      if (!phc_coeffs_Hx_file) { std::cerr << "Numerical PHC Hx file not found!\n"; exit(1); }
      else {
        while (phc_coeffs_Hx_file >> s_real >> s_imag) {
          (h_phc_coeffs_Hx)[d] = MAGMA_Z_MAKE(s_real, s_imag);
          d++;
        }
      }
      //> Ht
      d = 0;
      if (!phc_coeffs_Ht_file) { std::cerr << "Numerical PHC Ht file not found!\n"; exit(1); }
      else {
        while (phc_coeffs_Ht_file >> s_real >> s_imag) {
          (h_phc_coeffs_Ht)[d] = MAGMA_Z_MAKE(s_real, s_imag);
          d++;
        }
      }
    }

    read_success = (start_sols_read_success && start_coeffs_read_success && targetParams_read_success && Hx_file_read_success && Ht_file_read_success);

    //> write only the real solutions to files
    std::ofstream real_sols_file;
    std::string write_real_sols_file_dir = repo_root_dir;
    if (HC_problem == "3views_4pts")        { 
      if (RANSAC) {
        write_real_sols_file_dir = std::string("/users/cchien3/data/cchien3/bitbucket-repos/gpu-phc/auto-gen-tools/RANSAC/outlier_ratio_"); 
        write_real_sols_file_dir.append(std::to_string(ratio));
        write_real_sols_file_dir.append("/GPUHC_Solutions/"); 
      }
      else { write_real_sols_file_dir.append("problems/3views_4pts/synthetic_evaluation/GPUHC_Solutions_DP/"); }
    }
    else if (HC_problem == "six_lines_6x6") { write_real_sols_file_dir.append("problems/six_lines_6x6/synthetic_evaluation/GPUHC_Solutions_DP/"); }
    write_real_sols_file_dir.append(padded_fileName);
    real_sols_file.open(write_real_sols_file_dir);
    if ( !real_sols_file.is_open() ) { std::cout<<"Solutions for evaluation write files cannot be opened!"<<std::endl; exit(1); }

    //> Call homotopy continuation solver
    if (read_success) {
      magmaHCWrapperDP64::homotopy_continuation_solver(h_startSols, h_Track, h_startParams, h_targetParams, h_Hx_idx, h_Ht_idx, 
                                                   h_phc_coeffs_Hx, h_phc_coeffs_Ht, pp, HC_problem, real_sols_file, timings_file
                                                   );
    }
    else {
      std::cout<<"read files failed!"<<std::endl;
      exit(1);
    }

    real_sols_file.close();
  }

  delete pp;
  delete cm;
  magma_free_cpu( h_startSols );
  magma_free_cpu( h_Track );
  magma_free_cpu( h_startParams );
  magma_free_cpu( h_targetParams );
  magma_free_cpu( h_phc_coeffs_Hx );
  magma_free_cpu( h_phc_coeffs_Ht );

  magma_free_cpu( h_Hx_idx );
  magma_free_cpu( h_Ht_idx );

  timings_file.close();

  return 0;
}
