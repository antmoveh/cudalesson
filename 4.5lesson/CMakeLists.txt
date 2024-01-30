# 最低版本要求
cmake_minimum_required(VERSION 3.20)

# 项目信息
project(global_memory LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14) # 用于指定CUDA编译器应该使用的CUDA c++标准的版本
set(CMAKE_CUDA_STANDARD_REQUIRD ON) # 表明如果找不到指定版本的CUDA编译器，将发生错误
set(CMAKE_CXX_STANDARD 14) # 用于指定C++编译器应该使用的C++标准版本
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的C++编译器，将发生错误

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
    message(FATAL_ERROR "Cannot find CUDA")
endif()

# 添加可执行文件
add_executable(static_shared_memroy static_shared_memory.cu common.cuh)
include_directories(static_shared_memroy, PRIVATE ${CUDA_INCLUDE_DIRS})
# 链接库
target_link_libraries(static_shared_memroy PRIVATE ${CUDA_LIBRARIES})


# 添加可执行文件
add_executable(dynamic_shared_memroy static_shared_memory.cu common.cuh)
include_directories(dynamic_shared_memroy, PRIVATE ${CUDA_INCLUDE_DIRS})
# 链接库
target_link_libraries(dynamic_shared_memroy PRIVATE ${CUDA_LIBRARIES})