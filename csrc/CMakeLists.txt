add_library(cuda_operator_common INTERFACE)
target_include_directories(cuda_operator_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)

# Operator targets are added incrementally as implementations land.
