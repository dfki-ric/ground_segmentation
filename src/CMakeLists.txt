rock_library(pointcloud_object_detection
    SOURCES Dummy.cpp
    HEADERS Dummy.hpp)

rock_executable(pointcloud_object_detection_bin Main.cpp
    DEPS pointcloud_object_detection)

