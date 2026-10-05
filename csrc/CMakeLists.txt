add_library(cuda_operator_common INTERFACE)
target_include_directories(cuda_operator_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_library(cuda_operator_lab SHARED
  reduction/reduction.cu
  softmax/softmax.cu
  rmsnorm/rmsnorm.cu
  layernorm/layernorm.cu
  fused_residual_layernorm/fused_residual_layernorm.cu
  fused_bias_gelu/fused_bias_gelu.cu
  swiglu/swiglu.cu
  gemm_bias_gelu/gemm_bias_gelu.cu
  gemm_swiglu/gemm_swiglu.cu
)

target_include_directories(cuda_operator_lab
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/reduction
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax
    ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm
    ${CMAKE_CURRENT_SOURCE_DIR}/layernorm
    ${CMAKE_CURRENT_SOURCE_DIR}/fused_residual_layernorm
    ${CMAKE_CURRENT_SOURCE_DIR}/fused_bias_gelu
    ${CMAKE_CURRENT_SOURCE_DIR}/swiglu
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm_bias_gelu
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm_swiglu
)

target_link_libraries(cuda_operator_lab
  PRIVATE
    cuda_operator_common
    ${CMAKE_DL_LIBS}
)

set_target_properties(cuda_operator_lab PROPERTIES
  OUTPUT_NAME cuda_operator_lab
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  CUDA_ARCHITECTURES 89
)