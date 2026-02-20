cmake_minimum_required(VERSION 3.10)

find_package(GTest REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

enable_testing()

add_executable(test_pointcloud_grid
  test_pointcloud_grid.cpp
)

target_include_directories(test_pointcloud_grid PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_pointcloud_grid
  Eigen3::Eigen
  GTest::gtest
  GTest::gtest_main
  ${PCL_LIBRARIES}
)

add_test(
  NAME PointCloudGridTests
  COMMAND test_pointcloud_grid
)

add_executable(test_visual_segmentation
  test_visual_segmentation.cpp
)

target_include_directories(test_visual_segmentation PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_visual_segmentation
  Eigen3::Eigen
  ${PCL_LIBRARIES}
)