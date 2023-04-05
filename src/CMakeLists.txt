find_package(PCL 1.7 REQUIRED COMPONENTS io common filters kdtree segmentation)

rock_library(pointcloud_obstacle_detection
    SOURCES processPointClouds.cpp 
    HEADERS processPointClouds.h Box.h
    DEPS_PKGCONFIG
        vizkit3d_debug_drawings-commands
        base-types 
        pcl_io-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
        pcl_common-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
        pcl_filters-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
        pcl_kdtree-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
        pcl_segmentation-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
)

