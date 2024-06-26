#ifndef define_params_and_read_files_cu
#define define_params_and_read_files_cu
// ==============================================================================
//
// Modifications
//    Chien  22-10-31:   Initially Created (Copied from other repos)
//    Chien  24-03-02:   For 3-views 4-points problem, only 583 solutions are needed to find 
//                       the actual solution of the problem. This is the same as stated in the paper.
//                       The change is to define numOfTracks as 583 for 3views_4pts problem.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ==============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>

#include "magmaHC-problems.cuh"

namespace magmaHCWrapperDP64 {

  void print_usage()
  {
    std::cerr << "===================================================================================================================\n";
    std::cerr << "Usage: ./magmaHC-main <input-argument> <command>\n\n";
    std::cerr << "Choices of input arguments and commands\n";
    std::cerr << "<input-argument>      <command>\n"
                 "       -p             <problem>                  # (or --problem)  : minimal problem name \n"
                 "       -h                                        # (or --help)     : print this help message\n\n";
    std::cerr << "----------------------- NOTICE -----------------------\n";
    std::cerr << "1. Order matters.\n";
    std::cerr << "2. If <input-argument> and <command> are not specified, the help message will be automatically shown.\n\n";
    std::cerr << "----------------------- Examples -----------------------\n";
    std::cerr << "./magmaHC-main -p six_lines_6x6      # solve six lines minimal problem under a generalized camera model\n";
    std::cerr << "===================================================================================================================\n";
  }

  void problem_params::define_problem_params(std::string problem_filename, std::string HC_problem)
  {
    if (HC_problem == "six_lines_6x6") {

      //> problem specifications
      numOfParams = 36;
      numOfTracks = 600;
      numOfVars = 6;
      numOfCoeffsFromParams = 564;

      //> Indexing evaluations
      Hx_maximal_terms = 40;
      Hx_maximal_parts = 5;
      Ht_maximal_terms = 100;
      Ht_maximal_parts = 6;

      max_orderOf_t = 3;
    }
    else if (HC_problem == "3views_4pts") {

      //> problem specifications
      numOfParams = 45;
      numOfTracks = 583;
      numOfVars = 12;
      numOfCoeffsFromParams = 66;

      //> Indexing evaluations
      Hx_maximal_terms = 3;
      Hx_maximal_parts = 3;
      Ht_maximal_terms = 10;
      Ht_maximal_parts = 4;

      max_orderOf_t = 2;
    }
    else {
      std::cout<<"You are entering invalid HC problem in your input argument!"<<std::endl;
      print_usage();
      exit(1);
    }
  }
} // end of namespace

#endif
