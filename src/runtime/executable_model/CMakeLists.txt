cmake_minimum_required(VERSION 3.20)
project(executable_model)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

if(UNIT_TEST)
add_executable(executable_model_test
               executable_model_test.cc
               ${SRC_TOP}/common/enn_debug.cc
               ${SRC_TOP}/common/enn_utils.cc)
target_include_directories(executable_model_test PRIVATE ${SRC_TOP})
target_link_libraries(executable_model_test ${GTEST_LDFLAGS})
add_test(NAME executable_model_test COMMAND executable_model_test)

add_executable(executable_operator_list_test
                    executable_operator_list_test.cc
                    ${SRC_TOP}/common/enn_debug.cc
                    ${SRC_TOP}/common/enn_utils.cc)
target_include_directories(executable_operator_list_test PRIVATE ${SRC_TOP})
target_link_libraries(executable_operator_list_test ${GTEST_LDFLAGS})
add_test(NAME executable_operator_list_test COMMAND executable_operator_list_test)
endif()
