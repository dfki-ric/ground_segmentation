cmake_minimum_required(VERSION 3.9)
project(ground_segmentation
        VERSION 1.0
        DESCRIPTION "Ground segmentation in pointcloud")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "RELEASE")

add_subdirectory(src)
add_subdirectory(test)

find_package(Doxygen REQUIRED)

set(DOXYFILE_IN  ${CMAKE_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_BINARY_DIR}/Doxyfile)

configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
)