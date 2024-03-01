add_executable(test_ground_detection main.cpp)
target_link_libraries(test_ground_detection PUBLIC pointcloud_obstacle_detection)

# Install the executable
install(TARGETS test_ground_detection DESTINATION bin)
