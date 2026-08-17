cmake_minimum_required(VERSION 3.20)
project(cadencevad_portable C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# The shared runtime core lives beside the Apple backend; only the DSP layer
# differs between platforms.
set(CADENCEVAD_CORE
  ${CMAKE_CURRENT_SOURCE_DIR}/../macos/cadencevad_native.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../macos/cadencevad_weights.c
)

if(APPLE AND NOT CADENCEVAD_FORCE_PORTABLE)
  set(CADENCEVAD_DSP ${CMAKE_CURRENT_SOURCE_DIR}/fv_dsp_accelerate.c)
else()
  set(CADENCEVAD_DSP ${CMAKE_CURRENT_SOURCE_DIR}/fv_dsp_portable.c)
endif()

add_library(cadencevad SHARED ${CADENCEVAD_CORE} ${CADENCEVAD_DSP})
target_include_directories(cadencevad PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../macos
)

if(APPLE AND NOT CADENCEVAD_FORCE_PORTABLE)
  target_link_libraries(cadencevad PRIVATE "-framework Accelerate")
endif()

if(MSVC)
  # /fp:precise keeps the shipped numerics; /arch:AVX2 enables the intrinsic path.
  target_compile_options(cadencevad PRIVATE /O2 /fp:precise)
  if(CADENCEVAD_ENABLE_AVX2)
    target_compile_options(cadencevad PRIVATE /arch:AVX2)
    target_compile_definitions(cadencevad PRIVATE __FMA__)
  endif()
else()
  target_compile_options(cadencevad PRIVATE -O3 -ffp-contract=off)
  if(CADENCEVAD_ENABLE_AVX2)
    target_compile_options(cadencevad PRIVATE -mavx2 -mfma)
  endif()
  find_library(M_LIB m)
  if(M_LIB)
    target_link_libraries(cadencevad PRIVATE ${M_LIB})
  endif()
endif()

set_target_properties(cadencevad PROPERTIES
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(cadencevad_bench ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_portable.c)
target_link_libraries(cadencevad_bench PRIVATE cadencevad)
