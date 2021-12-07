cmake_minimum_required(VERSION 3.5)

PROJECT(yolov4-triton-tensorrt VERSION 1.1.0 LANGUAGES CXX)

# CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_75;code=sm_75)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()


# LAYER PLUGIN LIB
cuda_add_library(yolo416layerplugin SHARED ${PROJECT_SOURCE_DIR}/layers/yolo416layer.cu)
# needs to be linked here because triton will not have those libs preloaded!
target_link_libraries(yolo416layerplugin nvinfer cudart)

# EXECUTABLE
add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(main nvinfer cudart yolo416layerplugin)
