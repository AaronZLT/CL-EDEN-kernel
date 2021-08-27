cmake_minimum_required(VERSION 3.20)
project(tensor)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

if(UNIT_TEST)
add_executable(feature_map_test feature_map_test.cc)
target_include_directories(feature_map_test PRIVATE ${SRC_TOP})
target_link_libraries(feature_map_test ${GTEST_LDFLAGS})
add_test(NAME feature_map_test COMMAND feature_map_test)

add_executable(parameter_test parameter_test.cc)
target_include_directories(parameter_test PRIVATE ${SRC_TOP})
target_link_libraries(parameter_test ${GTEST_LDFLAGS})
add_test(NAME parameter_test COMMAND parameter_test)

add_executable(scalar_test scalar_test.cc)
target_include_directories(scalar_test PRIVATE ${SRC_TOP})
target_link_libraries(scalar_test ${GTEST_LDFLAGS})
add_test(NAME scalar_test COMMAND scalar_test)

endif()