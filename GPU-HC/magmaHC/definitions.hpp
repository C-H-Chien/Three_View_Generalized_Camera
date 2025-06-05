//> ======================================================
//> Macros definitions
//> ======================================================


#define WRITE_FILES_FOLDER                      std::string("Output_Write_Files/")

//> RANSAC Settings
#define NUM_OF_RANSAC_ITERATIONS                (1)
#define IMAG_PART_TOL                           (1e-5)
#define ROT_RESIDUAL_TOL                        (1e-1)
#define TRANSL_RESIDUAL_TOL                     (1e-1)
#define TEST_RANSAC_TIMES                       (1)
#define REPROJ_ERROR_INLIER_THRESH              (2) //> in pixels

//> Settings for GPU-HC Kernel
#define USE_SINGLE_PRECISION                    (false)

//> Evaluation macros
#define DUPLICATE_SOL_DIFF_TOL                  (1e-4)
#define DEBUG_EVALUATOR                         (false)
#define IS_SO3_DET_R_TOL                        (1e-5)

//> Define universal MACROS for both single and double precision
#if USE_SINGLE_PRECISION
    #define MAGMA_MAKE_COMPLEX(r,i)             MAGMA_C_MAKE(r,i)
    #define MAGMA_COMPLEX_ZERO                  MAGMA_C_ZERO
    #define MAGMA_COMPLEX_ONE                   MAGMA_C_ONE
    #define MAGMA_COMPLEX_DIV(a,b)              MAGMA_C_DIV(a,b)
    #define MAGMA_COMPLEX_REAL(r)               MAGMA_C_REAL(r)
    #define MAGMA_COMPLEX_IMAG(i)               MAGMA_C_IMAG(i)
    #define MAGMA_NUM_ZERO                      (0.0)
    #define MAGMA_NUM_ONE                       (1.0)
    #define ZERO_IMAG_PART_TOL                  (1e-4)              //> Used for evaluation: finding real solutions
#else
    #define MAGMA_MAKE_COMPLEX(r,i)             MAGMA_Z_MAKE(r,i)
    #define MAGMA_COMPLEX_ZERO                  MAGMA_Z_ZERO
    #define MAGMA_COMPLEX_ONE                   MAGMA_Z_ONE
    #define MAGMA_COMPLEX_DIV(a,b)              MAGMA_Z_DIV(a,b)
    #define MAGMA_COMPLEX_REAL(r)               MAGMA_Z_REAL(r)
    #define MAGMA_COMPLEX_IMAG(i)               MAGMA_Z_IMAG(i)
    #define MAGMA_NUM_ZERO                      (0.0)
    #define MAGMA_NUM_ONE                       (1.0)
    #define ZERO_IMAG_PART_TOL                  (1e-8)              //> Used for evaluation: finding real solutions
#endif

//> Settings for Debugging
#define SHOW_PROBLEM_SETTINGS                   (true)
#define GPU_DEBUG                               (true)
#define DATA_READER_DEBUG                       (false)
#define RANSAC_DEBUG                            (false)

#define SHARED_ALLOC_INIT(ptr) \
    char* __shared_alloc_ptr = (char*)(ptr);

#define SHARED_ALLOC(type, name, count) \
    __shared_alloc_ptr = (char*)(((uintptr_t)__shared_alloc_ptr + alignof(type) - 1) & ~(alignof(type) - 1)); \
    type* name = (type*)__shared_alloc_ptr; \
    __shared_alloc_ptr += sizeof(type) * (count);

//> [DO NOT CHANGE] The following macros are constant. They are used for shuffle operation in a warp level.
#define FULL_MASK                               (0xffffffff)
#define WARP_SIZE                               (32)
#define HALF_WARP_SIZE                          (16)

//> [DO NOT CHANGE] Constant values
#define ONE_OVER_SIX                            (0.166666666667)

//> CUDA error check
#define cudacheck( a )  do { \
                            cudaError_t e = a; \
                            if(e != cudaSuccess) { \
                                printf("\033[1;31m"); \
                                printf("Error in %s:%d %s\n", __func__, __LINE__, cudaGetErrorString(e)); \
                                printf("\033[0m"); \
                            }\
                        } while(0)

#define LOG_INFOR_MESG(info_msg)        printf("\033[1;32m[INFO] %s\033[0m\n", std::string(info_msg).c_str() );
#define LOG_FILE_ERROR(err_msg)         printf("\033[1;31m[ERROR] File %s not found!\033[0m\n", std::string(err_msg).c_str() );
#define LOG_ERROR(err_msg)              printf("\033[1;31m[ERROR] %s\033[0m\n", std::string(err_msg).c_str() );
#define LOG_DATA_LOAD_ERROR(err_msg)    printf("\033[1;31m[DATA LOAD ERROR] %s not loaded successfully!\033[0m\n", std::string(err_msg).c_str() );
#define LOG_PRINT_HELP_MESSAGE          printf("Usage: ./magmaHC-main [options] [path]\n\n" \
                                               "options:\n" \
                                               "  -h, --help        show this help message and exit\n" \
                                               "  -p, --directory   problem name, e.g. generalized_3views_4pts\n");


