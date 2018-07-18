# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# 项目信息
project (Demo)

# 添加 math 子目录
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
add_subdirectory(src)

find_package(CUDA)

file(GLOB CURRENT_SOURCES  *.cpp *.cu)
file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
source_group("Source" FILES ${CURRENT_SOURCES})
source_group("Include" FILES ${CURRENT_HEADERS})

# 指定生成目标
CUDA_ADD_EXECUTABLE(Demo ${CURRENT_HEADERS} ${CURRENT_SOURCES})

# 添加链接库
target_link_libraries(Demo zxy)



