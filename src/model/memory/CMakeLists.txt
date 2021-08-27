cmake_minimum_required(VERSION 3.20)
project(memory)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

if(UNIT_TEST)
add_executable(buffer_test buffer_test.cc)
target_include_directories(buffer_test PRIVATE ${SRC_TOP})
target_link_libraries(buffer_test ${GTEST_LDFLAGS})
add_test(NAME buffer_test COMMAND buffer_test)
endif()