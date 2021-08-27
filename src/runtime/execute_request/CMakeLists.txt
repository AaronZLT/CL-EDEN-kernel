cmake_minimum_required(VERSION 3.20)
project(execute_request)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

if(UNIT_TEST)
add_executable(execute_request_test
                    execute_request_test.cc
                    ${SRC_TOP}/common/enn_debug.cc
                    ${SRC_TOP}/common/enn_utils.cc)
target_include_directories(execute_request_test PRIVATE ${SRC_TOP})
target_link_libraries(execute_request_test ${GTEST_LDFLAGS})
add_test(NAME execute_request_test COMMAND execute_request_test)
endif()
