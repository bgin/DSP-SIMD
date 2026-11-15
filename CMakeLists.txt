cmake_minimum_required(VERSION 3.16)
project(DSP_SIMD VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(DSP_SIMD_ENABLE_AVX512 "Enable AVX-512 instruction set flags" ON)
option(DSP_SIMD_ENABLE_SVML "Enable Intel SVML dependent routines" OFF)
option(DSP_SIMD_BUILD_TESTS "Build unit test executables" ON)
option(DSP_SIMD_BUILD_PERF "Build performance benchmark executables" ON)

# Collect core C/C++ implementation sources
file(GLOB_RECURSE DSP_SIMD_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/basic_functions/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/basic_functions/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/complex_math/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/complex_math/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/linalg/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/linalg/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math_functions/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math_functions/*.cpp"
)

# Exclude SVML-specific source files if SVML is disabled
if(NOT DSP_SIMD_ENABLE_SVML)
    list(FILTER DSP_SIMD_SOURCES EXCLUDE REGEX ".*_svml_.*")
endif()

add_library(dsp_simd STATIC ${DSP_SIMD_SOURCES})

target_include_directories(dsp_simd PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/basic_functions"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/complex_math"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/linalg"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math_functions"
)

# Compiler optimization and SIMD ISA flags
if(MSVC)
    target_compile_options(dsp_simd PUBLIC /O2)
    if(DSP_SIMD_ENABLE_AVX512)
        target_compile_options(dsp_simd PUBLIC /arch:AVX512)
    else()
        target_compile_options(dsp_simd PUBLIC /arch:AVX2)
    endif()
else()
    target_compile_options(dsp_simd PUBLIC -O3 -Wall -Wextra)
    if(DSP_SIMD_ENABLE_AVX512)
        target_compile_options(dsp_simd PUBLIC -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma)
    else()
        target_compile_options(dsp_simd PUBLIC -mavx2 -mfma)
    endif()
endif()

# Link standard math library on Unix-like systems
if(UNIX)
    target_link_libraries(dsp_simd PUBLIC m)
endif()

# Unit Tests
if(DSP_SIMD_BUILD_TESTS)
    enable_testing()
    file(GLOB UNIT_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/unit-tests/*.cpp")
    foreach(test_src ${UNIT_TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE dsp_simd)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Performance Tests
if(DSP_SIMD_BUILD_PERF)
    file(GLOB PERF_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/perf-tests/*.cpp")
    foreach(perf_src ${PERF_TEST_SOURCES})
        get_filename_component(perf_name ${perf_src} NAME_WE)
        add_executable(${perf_name} ${perf_src})
        target_link_libraries(${perf_name} PRIVATE dsp_simd)
    endforeach()
endif()

