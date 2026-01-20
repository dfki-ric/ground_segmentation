cmake_minimum_required(VERSION 3.9)
project(ground_segmentation
        VERSION 1.0
        DESCRIPTION "Ground segmentation in pointcloud")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "RELEASE")

add_subdirectory(src)

# Create and install the version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}-config-version.cmake"
  VERSION ${GROUND_SEGMENTATION_VERSION}
        COMPATIBILITY SameMajorVersion
)

install(
        FILES
                ${PROJECT_NAME}-config.cmake
                ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
        DESTINATION
    lib/cmake/${PROJECT_NAME}
)

add_subdirectory(test)