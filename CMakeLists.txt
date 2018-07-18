# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)

# 项目信息
project (Demo)

# 添加 math 子目录
add_subdirectory(src)

# 指定生成目标
add_executable(Demo main.cpp)

# 添加链接库
target_link_libraries(Demo zxy)
