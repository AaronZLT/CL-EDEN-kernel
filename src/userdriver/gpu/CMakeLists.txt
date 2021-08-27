cmake_minimum_required(VERSION 3.20)
project(enn_gpu_ud)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

file(GLOB operator_files
    operators/*.cpp
)

set(_INCLUDE_THIRD_PARTY userdriver/gpu/third_party/opencl_stub/CL
                         userdriver/gpu/third_party/opencl_stub
)
set(source_files
    gpu_userdriver.cc
    gpu_op_constructor.cc
    gpu_op_executor.cc
    common/CLComputeLibrary.cpp
)

add_library(enn_user_driver_gpu SHARED ${source_files} ${operator_files})
target_include_directories(enn_user_driver_gpu PRIVATE ${SRC_TOP})

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
add_library(enn_memory_manager SHARED ${SRC_TOP}/common/enn_memory_manager.cc ${SRC_TOP}/common/enn_memory_allocator_heap.cc)
target_include_directories(enn_memory_manager PRIVATE ${SRC_TOP})
add_library(generator SHARED ${SRC_TOP}/model/generator/generator.cc)
target_include_directories(generator PRIVATE ${SRC_TOP} ${_INCLUDE_THIRD_PARTY})
add_library(model_pool_manager SHARED ${SRC_TOP}/model/pool/manager.cc)
target_include_directories(model_pool_manager PRIVATE ${SRC_TOP})
endif()

if(NNC_V1)
    add_definitions(-DSCHEMA_NNC_V1)
    message("-- SCHEMA_VERSION: nnc_v1")
endif()

if(UNIT_TEST)
add_executable(gpu_userdriver_test gpu_userdriver_test.cc)
target_include_directories(gpu_userdriver_test PRIVATE ${SRC_TOP} ${_INCLUDE_THIRD_PARTY})
target_link_libraries(gpu_userdriver_test enn_user_driver_gpu enn_dbg_utils enn_memory_manager generator model_pool_manager ${GTEST_LDFLAGS})
add_test(NAME gpu_userdriver_test COMMAND gpu_userdriver_test)

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
