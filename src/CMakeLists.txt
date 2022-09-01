find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

rock_library(pointcloud_object_detection
    SOURCES processPointClouds.cpp
    HEADERS processPointClouds.h
            render/box.h)

