# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.20)
project(MixedOffload LANGUAGES HIP CXX)

find_package(OpenMP REQUIRED)

add_library(hip_kernels STATIC hip.cpp)
set_source_files_properties(hip.cpp PROPERTIES LANGUAGE HIP)

add_library(omp_kernels STATIC omp.cpp)
target_compile_options(omp_kernels PRIVATE -fopenmp --offload-arch=gfx942)

add_executable(mixed main.cpp)
target_link_libraries(mixed PRIVATE hip_kernels omp_kernels OpenMP::OpenMP_CXX)
target_link_options(mixed PRIVATE -fopenmp --offload-arch=gfx942)
