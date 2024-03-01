# CMakeLists.txt has to be located in the project folder and cmake has to be
# executed from 'project/build' with 'cmake ../'.
cmake_minimum_required(VERSION 3.9)
project(pointcloud_obstacle_detection
        VERSION 1.0
        DESCRIPTION "Obstacle detection in pointcloud")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "RELEASE")

add_subdirectory(src)

# Create and install the version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}-config-version.cmake"
	VERSION ${POINTCLOUD_OBSTACLE_DETECTION_VERSION}
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

# Install package-manifest
#install(
#        FILES package.xml
#        DESTINATION ${CMAKE_INSTALL_DIR}/share/${PROJECT_NAME}
#)
