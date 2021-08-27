cmake_minimum_required(VERSION 3.20)
project(enn_cpu_ud)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

if(NNC_V1)
    add_definitions(-DSCHEMA_NNC_V1)
    message("-- SCHEMA_VERSION: nnc_v1")
endif()

if(MEM_DUMP)
    add_definitions(-D__FORCE_DUMP__)
    message("-- __FORCE_DUMP__: enable")
endif()

file(GLOB operator_files
    operators/*.cpp
    operators/neon_impl/BboxUtil_batch_single.cpp
    operators/neon_impl/BboxUtil.cpp
    operators/neon_impl/DetectionOutput.cpp
    operators/neon_impl/NormalizedBbox.cpp
)

set(source_files
    cpu_userdriver.cc
    cpu_op_constructor.cc
    cpu_op_executor.cc
    common/NEONComputeLibrary.cpp
)

add_library(enn_user_driver_cpu SHARED ${source_files} ${operator_files})
target_include_directories(enn_user_driver_cpu PRIVATE ${SRC_TOP})

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
# Build enn_memory_manager
add_library(enn_memory_manager SHARED ${SRC_TOP}/common/enn_memory_manager.cc ${SRC_TOP}/common/enn_memory_allocator_heap.cc)
target_include_directories(enn_memory_manager PRIVATE ${SRC_TOP})
# Build enn_parser
set(parser_source_files
    ${SRC_TOP}/model/parser/parser.cc
    ${SRC_TOP}/model/parser/strategy/nnc_parse_strategy.cc
    ${SRC_TOP}/model/parser/strategy/cgo_parse_strategy.cc
    ${SRC_TOP}/model/parser/strategy/coded_parse_strategy.cc)
add_library(enn_parser SHARED ${parser_source_files})
target_include_directories(enn_parser PRIVATE ${SRC_TOP})
# Build enn_raw_model
add_library(enn_raw_model SHARED ${SRC_TOP}/model/raw/model.cc)
target_include_directories(enn_raw_model PRIVATE ${SRC_TOP})
endif()

if(NNC_V1)
    add_definitions(-DSCHEMA_NNC_V1)
    message("-- SCHEMA_VERSION: nnc_v1")
endif()

if(UNIT_TEST)
add_executable(cpu_userdriver_test cpu_userdriver_test.cc)
target_include_directories(cpu_userdriver_test PRIVATE ${SRC_TOP})
target_link_libraries(cpu_userdriver_test enn_user_driver_cpu
                      enn_dbg_utils enn_memory_manager enn_parser enn_raw_model ${GTEST_LDFLAGS})
add_test(NAME cpu_userdriver_test COMMAND cpu_userdriver_test)

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
