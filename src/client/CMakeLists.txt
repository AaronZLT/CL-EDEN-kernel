cmake_minimum_required(VERSION 3.20)
project(client)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

if(UNIT_TEST)
add_executable(enn_model_container_test enn_model_container_test.cc)
target_include_directories(enn_model_container_test PRIVATE ${SRC_TOP})
target_link_libraries(enn_model_container_test enn_dbg_utils ${GTEST_LDFLAGS})
add_test(NAME enn_model_container_test COMMAND enn_model_container_test)
endif()