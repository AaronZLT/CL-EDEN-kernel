cmake_minimum_required(VERSION 3.20)
project(enn_parser)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

if(NNC_V1)
    add_definitions(-DSCHEMA_NNC_V1)
    message("-- SCHEMA_VERSION: nnc_v1")
endif()

set(source_files
    parser.cc
    strategy/nnc_parse_strategy.cc
    strategy/cgo_parse_strategy.cc
    strategy/coded_parse_strategy.cc)

add_library(enn_parser      SHARED  ${source_files})
target_include_directories(enn_parser PRIVATE ${SRC_TOP})
add_library(enn_raw_model   SHARED  ${SRC_TOP}/model/raw/model.cc)
target_include_directories(enn_raw_model PRIVATE ${SRC_TOP})

if(UNIT_TEST)

# Build enn_memory_manager (for parser unit test)
# 1. check if build is local build.
# 2. check if build is from model.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR ${CMAKE_SOURCE_DIR}/parser STREQUAL CMAKE_CURRENT_SOURCE_DIR)
add_library(enn_memory_manager SHARED ${SRC_TOP}/common/enn_memory_manager.cc ${SRC_TOP}/common/enn_memory_allocator_heap.cc)
target_include_directories(enn_memory_manager PRIVATE ${SRC_TOP})
endif()

add_executable(parser_test parser_test.cc)
target_include_directories(parser_test PRIVATE ${SRC_TOP})
target_link_libraries(parser_test enn_parser enn_raw_model enn_dbg_utils enn_memory_manager ${GTEST_LDFLAGS})
add_test(NAME parser_test COMMAND parser_test)

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
