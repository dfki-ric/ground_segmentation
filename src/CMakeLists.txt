cmake_minimum_required(VERSION 3.16)
project(ground_segmentation VERSION 0.1.0 LANGUAGES CXX)

# -------------------------------------------------------------------
# Dependencies (BUILD time only)
# -------------------------------------------------------------------

find_package(PCL REQUIRED COMPONENTS
  common
  io
  filters
  kdtree
  segmentation
  visualization
)

find_package(Boost REQUIRED COMPONENTS serialization filesystem system)
find_package(VTK REQUIRED)
find_package(nanoflann REQUIRED)

# -------------------------------------------------------------------
# Interface library
# -------------------------------------------------------------------

add_library(${PROJECT_NAME} INTERFACE)

add_library(${PROJECT_NAME}::${PROJECT_NAME}
  ALIAS ${PROJECT_NAME}
)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_definitions(${PROJECT_NAME} INTERFACE
  NUMERIC_DEPRECATE=1
)

# -------------------------------------------------------------------
# Link dependencies (IMPORTED TARGETS ONLY)
# -------------------------------------------------------------------

target_link_libraries(${PROJECT_NAME} INTERFACE
  Boost::serialization
  Boost::filesystem
  Boost::system
  ${PCL_LIBRARIES}
  ${VTK_LIBRARIES}
  nanoflann::nanoflann
)

# -------------------------------------------------------------------
# Install headers
# -------------------------------------------------------------------


# ------------------------------------------------------------
# Install headers (REQUIRED for INTERFACE libraries)
# ------------------------------------------------------------

install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/include/
  DESTINATION include
)

# -------------------------------------------------------------------
# Export targets
# -------------------------------------------------------------------

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
)

install(
  EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION lib/cmake/${PROJECT_NAME}
)

# -------------------------------------------------------------------
# Package config files
# -------------------------------------------------------------------

include(CMakePackageConfigHelpers)

configure_package_config_file(
  cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION lib/cmake/${PROJECT_NAME}
)

set(PKG_CONFIG_DEPS
    pcl_io-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
    pcl_common-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
    pcl_filters-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
    pcl_kdtree-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
    pcl_segmentation-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
    pcl_visualization-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}
)

list(JOIN PKG_CONFIG_DEPS " " PKGCONFIG_REQUIRES)
set(PKGCONFIG_CFLAGS)

configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc
  DESTINATION lib/pkgconfig
)