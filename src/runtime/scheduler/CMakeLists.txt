cmake_minimum_required(VERSION 3.20)
project(scheduler)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../..)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

if(UNIT_TEST)
set(source_file
    ${SRC_TOP}/model/generator/generator.cc
    ${SRC_TOP}/model/raw/model.cc
    ${SRC_TOP}/model/parser/parser.cc
    ${SRC_TOP}/model/parser/strategy/nnc_parse_strategy.cc
    ${SRC_TOP}/model/parser/strategy/cgo_parse_strategy.cc
    ${SRC_TOP}/model/parser/strategy/coded_parse_strategy.cc
    ${SRC_TOP}/common/enn_debug.cc
    ${SRC_TOP}/common/enn_utils.cc
)

set(MODEL_LIB enn_model_)
add_library(${MODEL_LIB} SHARED ${source_file})
target_include_directories(${MODEL_LIB} PRIVATE ${SRC_TOP})

add_executable(static_scheduler_test static_scheduler_test.cc)
target_include_directories(static_scheduler_test PRIVATE ${SRC_TOP})
target_link_libraries(static_scheduler_test ${MODEL_LIB} ${GTEST_LDFLAGS})
add_test(NAME static_scheduler_test COMMAND static_scheduler_test)

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

