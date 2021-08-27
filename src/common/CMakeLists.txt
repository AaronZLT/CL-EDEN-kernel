cmake_minimum_required(VERSION 3.10)
project(common)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

add_library(enn_memory_manager SHARED enn_memory_manager.cc enn_memory_allocator_heap.cc)
target_include_directories(enn_memory_manager PRIVATE ${SRC_TOP})

if(UNIT_TEST)
add_executable(enn_debug_test enn_debug_test.cc)
target_include_directories(enn_debug_test PRIVATE ${SRC_TOP})
target_link_libraries(enn_debug_test enn_dbg_utils ${GTEST_LDFLAGS})
add_test(NAME enn_debug_test COMMAND enn_debug_test)

add_executable(enn_utils_test enn_utils_test.cc)
target_include_directories(enn_utils_test PRIVATE ${SRC_TOP})
target_link_libraries(enn_utils_test enn_dbg_utils ${GTEST_LDFLAGS})
add_test(NAME enn_utils_test COMMAND enn_utils_test)

add_executable(enn_utils_buffer_test enn_utils_buffer_test.cc)
target_include_directories(enn_utils_buffer_test PRIVATE ${SRC_TOP})
target_link_libraries(enn_utils_buffer_test enn_dbg_utils ${GTEST_LDFLAGS})
add_test(NAME enn_utils_buffer_test COMMAND enn_utils_buffer_test)

add_executable(enn_preference_generator_test enn_preference_generator_test.cc)
target_include_directories(enn_preference_generator_test PRIVATE ${SRC_TOP})
target_link_libraries(enn_preference_generator_test enn_dbg_utils ${GTEST_LDFLAGS})
add_test(NAME enn_preference_generator_test COMMAND enn_preference_generator_test)

set(exec_source_file enn_memory_manager_test.cc)
add_executable(enn_memory_manager_test enn_memory_manager_test.cc)
target_include_directories(enn_memory_manager_test PRIVATE ${SRC_TOP})
target_link_libraries(enn_memory_manager_test enn_memory_manager enn_dbg_utils ${GTEST_LDFLAGS})
add_test(NAME enn_memory_manager_test COMMAND enn_memory_manager_test)
endif()
