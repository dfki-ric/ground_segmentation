cmake_minimum_required(VERSION 3.9)
project(ground_segmentation
        VERSION 1.0
        DESCRIPTION "Ground segmentation in pointcloud")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "RELEASE")

add_subdirectory(src)
add_subdirectory(test)