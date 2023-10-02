
set (control_sources
  #>  input info
  define_params_and_read_files.cu

  #>  main gpu code
  magmaHC-problems.cuh
  homotopy_continuation_solver.cu

  #>  magma kernels
  gpu-kernels/magmaHC-kernels.h
  gpu-kernels/kernel_HC_Solver_six_lines_6x6.cu
  gpu-kernels/kernel_HC_Solver_3views_4pts.cu

  #>  device function
  dev-zgesv-batched-small.cuh
  dev-get-new-data.cuh
  gpu-idx-evals/dev-eval-indxing-six_lines_6x6.cuh
  gpu-idx-evals/dev-eval-indxing-3views_4pts.cuh

  #>  p2c constant matrix
  const-matrices/p2c-symbolic_3views_4pts.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories( "/users/cchien3/data/cchien3/magma-2.5.4/include" )
include_directories( "/users/cchien3/data/cchien3/magma-2.5.4/testing" )
include_directories( "/users/cchien3/data/cchien3/magma/control" )
include_directories( "/users/cchien3/data/cchien3/magma/magmablas" )
add_library( magmaHC-DP64 ${control_sources} )
target_link_libraries(magmaHC-DP64 
	PUBLIC OpenMP::OpenMP_CXX
        -L/gpfs/runtime/opt/magma-gpu/2.4.0/lib magma
        -L${CUDAPATH}/lib cublas cudart cusparse
        -L/gpfs/runtime/opt/openblas/0.3.7/lib openblas
        -L/usr/lib64 pthread
        )
set_target_properties(magmaHC-DP64 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

