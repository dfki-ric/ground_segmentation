find_package(PCL 1.7 REQUIRED COMPONENTS io common filters kdtree segmentation visualization)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(VTK REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(base_types REQUIRED IMPORTED_TARGET base-types)

set(SOURCES ground_detection.cpp processPointClouds.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    PkgConfig::base_types
    Boost::system
    Boost::filesystem
    Boost::serialization
    ${PCL_LIBRARIES}
    ${VTK_LIBRARIES}
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
)

target_compile_definitions(${PROJECT_NAME}
  PUBLIC
    NUMERIC_DEPRECATE=1
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
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
	DESTINATION cmake/${PROJECT_NAME}
)

#add_subdirectory(test)

set(PKGCONFIG_CFLAGS)
set(PKGCONFIG_REQUIRES
    base-types
)

configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc
	DESTINATION lib/pkgconfig
)
