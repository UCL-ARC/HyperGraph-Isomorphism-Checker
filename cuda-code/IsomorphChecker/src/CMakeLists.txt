cmake_minimum_required (VERSION 3.18)
project(HyperGraphCUDA LANGUAGES CXX CUDA  )

set(DEFAULT_BUILD_TYPE "Release")


# Separable compilation is required
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


find_package(CUDA REQUIRED)

set (CMAKE_CXX_COMPILER g++)
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)


file(GLOB_RECURSE ALL_SOURCES "*.cu" "*.cpp" "*.c")


add_executable(HyperGraphCUDA ${ALL_SOURCES})
target_compile_options(HyperGraphCUDA PRIVATE  -fPIC -use_fast_math -fconcepts -static  -g )
target_link_libraries(HyperGraphCUDA PRIVATE  -lstdc++fs)


target_include_directories(HyperGraphCUDA PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


set_target_properties(HyperGraphCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

set_property(TARGET HyperGraphCUDA PROPERTY CUDA_ARCHITECTURES 61 70 75 80 86 89 )
