cmake_minimum_required(VERSION 3.20)
project(runtime)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

# TODO(hj2020.kim): auto generatation of types.hal.h in CMakeLists.txt
# TODO(daewhan.kim): Enable engine test without ud

# flatbuffer headers uses #include by relative path
# include_directories(../ model/schema/)

set(source_file
    ${SRC_TOP}/model/generator/generator.cc
    ${SRC_TOP}/model/raw/model.cc
    ${SRC_TOP}/model/parser/parser.cc
    ${SRC_TOP}/model/parser/strategy/nnc_parse_strategy.cc
    ${SRC_TOP}/model/parser/strategy/cgo_parse_strategy.cc
    ${SRC_TOP}/model/parser/strategy/coded_parse_strategy.cc
    ${SRC_TOP}/runtime/pool/manager.cc
    ${SRC_TOP}/common/enn_debug.cc
    ${SRC_TOP}/common/enn_utils.cc
)

set(MODEL_LIB runtime)
add_library(${MODEL_LIB} SHARED ${source_file})
target_include_directories(${MODEL_LIB} PRIVATE ${SRC_TOP})

add_subdirectory(scheduler)
add_subdirectory(executable_model)
add_subdirectory(execute_request)
add_subdirectory(pool)
add_subdirectory(client_process)

add_library(engine SHARED engine.cc)
target_include_directories(engine PRIVATE ${SRC_TOP})
target_link_libraries(engine ${MODEL_LIB})


# if(UNIT_TEST)  # Disable until UD mock-up is enabled
# add_executable(engine_test ${source_file} engine_test.cc)
# target_include_directories(engine_test PRIVATE ${SRC_TOP} .)
# target_link_libraries(engine_test ${RUNTIME_LIB} ${MODEL_LIB} ${GTEST_LDFLAGS})
# add_test(NAME engine_test COMMAND engine_test)

# set(TEST_DATA_PATH ${CMAKE_CURRENT_BINARY_DIR}/test_data)
# file(MAKE_DIRECTORY ${TEST_DATA_PATH})
# file(GLOB FILES "${SRC_TOP}/../materials/models/*")
# message(STATUS "${FILES}")
# foreach(file IN LISTS FILES)
#     get_filename_component(filename ${file} NAME)
#     file(COPY ${file} DESTINATION ${TEST_DATA_PATH})
#     message(STATUS "${file}")
# endforeach()

# endif()
