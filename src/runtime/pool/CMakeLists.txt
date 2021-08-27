cmake_minimum_required(VERSION 3.20)
project(pool)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

add_library(model_pool_manager SHARED manager.cc)
target_include_directories(model_pool_manager PRIVATE ${SRC_TOP})

if(UNIT_TEST)
add_executable(pool_manager_test
                                pool_manager_test.cc
                                ${SRC_TOP}/common/enn_debug.cc
                                ${SRC_TOP}/common/enn_utils.cc)
target_include_directories(pool_manager_test PRIVATE ${SRC_TOP})
target_link_libraries(pool_manager_test model_pool_manager enn_dbg_utils ${GTEST_LDFLAGS})
add_test(NAME pool_manager_test COMMAND pool_manager_test)
endif()
