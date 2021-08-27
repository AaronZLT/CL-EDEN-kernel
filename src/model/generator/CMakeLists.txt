cmake_minimum_required(VERSION 3.20)
project(generator)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../..)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

add_library(generator SHARED generator.cc)
target_include_directories(generator PRIVATE ${SRC_TOP})

if(UNIT_TEST)

# Build enn_memory_manager (for parser unit test)
# 1. check if build is local build.
# 2. check if build is from model.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR ${CMAKE_SOURCE_DIR}/parser STREQUAL CMAKE_CURRENT_SOURCE_DIR)
add_library(enn_memory_manager SHARED ${SRC_TOP}/common/enn_memory_manager.cc ${SRC_TOP}/common/enn_memory_allocator_heap.cc)
target_include_directories(enn_memory_manager PRIVATE ${SRC_TOP})
endif()

set(parser_src
    ${SRC_TOP}/model/parser/parser.cc
    ${SRC_TOP}/model/parser/strategy/nnc_parse_strategy.cc
    ${SRC_TOP}/model/parser/strategy/cgo_parse_strategy.cc
    ${SRC_TOP}/model/parser/strategy/coded_parse_strategy.cc)
add_library(parser_ SHARED ${parser_src})
target_include_directories(parser_ PRIVATE ${SRC_TOP})

add_library(raw_model_ SHARED ${SRC_TOP}/model/raw/model.cc)
target_include_directories(raw_model_ PRIVATE ${SRC_TOP})

add_executable(generator_test generator_test.cc)
target_include_directories(generator_test PRIVATE ${SRC_TOP})
target_link_libraries(generator_test generator parser_ raw_model_ enn_dbg_utils enn_memory_manager ${GTEST_LDFLAGS})
add_test(NAME generator_test COMMAND generator_test)

set(TEST_DATA_PATH ${CMAKE_CURRENT_BINARY_DIR}/test_data)
file(MAKE_DIRECTORY ${TEST_DATA_PATH})
file(GLOB FILES "${SRC_TOP}/../materials/models/*")
message(STATUS "${FILES}")
foreach(file IN LISTS FILES)
    get_filename_component(filename ${file} NAME)
    file(COPY ${file} DESTINATION ${TEST_DATA_PATH})
    message(STATUS "${file}")
endforeach()

endif()

