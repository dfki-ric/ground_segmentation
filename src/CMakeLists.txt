find_package(PCL 1.2 REQUIRED)

rock_library(pointcloud_obstacle_detection
    SOURCES processPointClouds.cpp
    HEADERS processPointClouds.h Box.h
    DEPS_PKGCONFIG vizkit3d_debug_drawings
)

target_include_directories(pointcloud_obstacle_detection
PUBLIC
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(pointcloud_obstacle_detection
    ${PCL_LIBRARIES}
)
