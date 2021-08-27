cmake_minimum_required(VERSION 3.20)
project(operatpr_list)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

if(UNIT_TEST)
add_executable(operator_list_test operator_list_test.cc)
target_include_directories(operator_list_test PRIVATE ${SRC_TOP})
target_link_libraries(operator_list_test ${GTEST_LDFLAGS} enn_dbg_utils)
add_test(NAME operator_list_test COMMAND operator_list_test)

add_executable(operator_test operator_test.cc)
target_include_directories(operator_test PRIVATE ${SRC_TOP})
target_link_libraries(operator_test ${GTEST_LDFLAGS})
add_test(NAME operator_test COMMAND operator_test)
endif()