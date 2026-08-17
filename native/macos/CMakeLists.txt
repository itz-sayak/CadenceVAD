cmake_minimum_required(VERSION 3.24)
project(cadencevad_native C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT APPLE)
  message(FATAL_ERROR "The Accelerate runtime is available only on Apple platforms")
endif()

set(CADENCEVAD_WEIGHTS_DIR "" CACHE PATH "Directory containing generated weights")
if(NOT EXISTS "${CADENCEVAD_WEIGHTS_DIR}/cadencevad_weights.c")
  message(FATAL_ERROR "Set CADENCEVAD_WEIGHTS_DIR to an export-native output directory")
endif()

add_library(cadencevad_native STATIC
  cadencevad_native.c
  "${CMAKE_CURRENT_SOURCE_DIR}/../portable/fv_dsp_accelerate.c"
  "${CADENCEVAD_WEIGHTS_DIR}/cadencevad_weights.c"
)
target_include_directories(cadencevad_native PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../portable"
  "${CADENCEVAD_WEIGHTS_DIR}"
)
target_compile_options(cadencevad_native PRIVATE -O3)
target_compile_definitions(cadencevad_native PRIVATE ACCELERATE_NEW_LAPACK)
target_link_libraries(cadencevad_native PUBLIC "-framework Accelerate")

add_executable(cadencevad_benchmark benchmark.c)
target_link_libraries(cadencevad_benchmark PRIVATE cadencevad_native)
