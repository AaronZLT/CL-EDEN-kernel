cmake_minimum_required(VERSION 3.20)
project(graph)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

# add_subdirectory(iterator/methods)

if(UNIT_TEST)
add_executable(graph_test graph_test.cc)
target_include_directories(graph_test PRIVATE ${SRC_TOP})
target_link_libraries(graph_test ${GTEST_LDFLAGS} enn_dbg_utils)
add_test(NAME graph_test COMMAND graph_test)
endif()