cmake_minimum_required(VERSION 3.20)
project(model)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

set(source_file raw/model.cc)

add_subdirectory(generator)
add_subdirectory(parser)
# add_subdirectory(raw)  # Use raw/model.cc instead of add_subdirectory

add_library(enn_model SHARED ${source_file})
target_include_directories(enn_model PRIVATE ${SRC_TOP})
target_link_libraries(enn_model generator enn_parser )

if(UNIT_TEST)  # The followings have only header and test source code
add_executable(model_test
                    model_test.cc
                    ${SRC_TOP}/common/enn_debug.cc
                    ${SRC_TOP}/common/enn_utils.cc)
target_include_directories(model_test PRIVATE ${SRC_TOP})
target_link_libraries(model_test ${GTEST_LDFLAGS})
add_test(NAME model_test COMMAND model_test)

add_subdirectory(component/tensor)
add_subdirectory(component/operator)
add_subdirectory(graph)
add_subdirectory(memory)
endif()