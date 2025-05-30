
set (control_sources

  GPU_HC_Solver.hpp
  GPU_HC_Solver.cpp

  #> Read problem data
  Data_Reader.hpp
  Data_Reader.cpp

  #> Evaluations on HC solutions
  Evaluations.hpp
  Evaluations.cpp

  #>  magma kernels
  gpu-kernels/magmaHC-kernels.hpp
  gpu-kernels/kernel_HC_Solver_generalized_3views_4pts.cu
  gpu-kernels/kernel_HC_Solver_generalized_3views_6lines.cu

  #>  device function
  dev-cgesv-batched-small.cuh
  dev-cgesv-batched-small-sm.cuh
  dev-get-new-data.cuh
  gpu-idx-evals/dev-eval-indxing-generalized_3views_4pts.cuh
  gpu-idx-evals/dev-eval-indxing-generalized_3views_6lines.cuh

  #>  P2C
  PHC_Coeffs/p2c-generalized_3views_4pts.h
  PHC_Coeffs/p2c-numerical_generalized_3views_6lines.h

  #> Utilities
  util.hpp
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories( "/users/cchien3/data/cchien3/magma-2.5.4/include" )
include_directories( "/users/cchien3/data/cchien3/magma-2.5.4/control" )
include_directories( "/users/cchien3/data/cchien3/magma-2.5.4/magmablas" )

add_library( magmaHC ${control_sources} )
target_link_libraries(magmaHC
        -L/users/cchien3/data/cchien3/magma-cchien3/lib magma
        -L/gpfs/runtime/opt/openblas/0.3.23-u6k5fey/lib openblas
        -L/gpfs/data/bkimia/cchien3/yaml-cpp/bin/lib64 yaml-cpp
        )
set_target_properties(magmaHC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

