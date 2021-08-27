cmake_minimum_required(VERSION 3.20)
project(client_process)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

if(UNIT_TEST)
add_executable(client_process_test
                    client_process_test.cc
                    ${SRC_TOP}/common/enn_debug.cc
                    ${SRC_TOP}/common/enn_utils.cc)
target_include_directories(client_process_test PRIVATE ${SRC_TOP})
target_link_libraries(client_process_test ${GTEST_LDFLAGS})
add_test(NAME client_process_test COMMAND client_process_test)
endif()
