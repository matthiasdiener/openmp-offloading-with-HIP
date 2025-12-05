cmake_minimum_required(VERSION 3.20)
project(MixedOffload LANGUAGES HIP CXX)

set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm")

find_package(OpenMP REQUIRED)

add_library(hip_kernels STATIC hip.cpp)

set_source_files_properties(hip.cpp PROPERTIES LANGUAGE HIP)

add_library(omp_kernels STATIC omp.cpp)

target_compile_options(omp_kernels PRIVATE -fopenmp --offload-arch=gfx942)

add_executable(mixed main.cpp)

target_link_directories(mixed PUBLIC "${ROCM_PATH}/lib")

target_link_libraries(mixed PRIVATE hip_kernels omp_kernels OpenMP::OpenMP_CXX)

target_link_options(mixed PRIVATE -fopenmp --offload-arch=gfx942)
