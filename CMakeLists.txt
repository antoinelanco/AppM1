cmake_minimum_required(VERSION 3.5)
project(AppM1 C CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

file(GLOB_RECURSE SRC "src/*.h" "src/*.cpp")

if(CUDA_FOUND)
    file(GLOB_RECURSE CUDA_SRC  "./src/cuda/*.cu")
    list(APPEND CUDA_NVCC_FLAGS "-O3")
    list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
    #list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30; -std=c++11")
    cuda_compile(cuda_objs ${CUDA_SRC})
    set(SRC ${SRC} ${cuda_objs})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(AppM1 ${SRC})
target_link_libraries(AppM1 ${CUDA_LIBRARIES})