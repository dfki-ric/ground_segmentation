find_package(PCL REQUIRED COMPONENTS io common filters kdtree segmentation visualization)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(VTK REQUIRED)
find_package(nanoflann REQUIRED)

set(SOURCES pointcloud_processor.cpp ground_detection.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME} PUBLIC
    Boost::system
    Boost::filesystem
    Boost::serialization
    ${PCL_LIBRARIES}
    ${VTK_LIBRARIES}
    nanoflann::nanoflann
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
    NUMERIC_DEPRECATE=1
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
	    DESTINATION include/${PROJECT_NAME}
	    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

# Export the library interface
install(EXPORT ${PROJECT_NAME}-targets
        NAMESPACE ${PROJECT_NAME}::
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
