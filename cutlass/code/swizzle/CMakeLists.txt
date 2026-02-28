cmake_minimum_required(VERSION 3.20)

project(swizzle LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(main src/main.cu)
    