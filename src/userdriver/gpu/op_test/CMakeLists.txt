cmake_minimum_required(VERSION 3.20)
project(enn_gpu_ud_operator_test)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

include_directories(${SRC_TOP})
include_directories(${SRC_TOP}/userdriver/gpu/third_party/opencl_stub/CL)
include_directories(${SRC_TOP}/userdriver/gpu/third_party/opencl_stub)


set(LIBRARY_FILES
    enn_dbg_utils
    ${GTEST_MAIN_LDFLAGS}
    ${GTEST_LDFLAGS})

enable_testing()
#-------------------------------------------------------------------#
# Linux(emulator) build is possible for only operator not used CL #
#-------------------------------------------------------------------#

set(SOURCE_FILES CLNormalization_test.cpp ../operators/CLNormalization.cpp)
add_executable(enn_gpu_op_CLNormalization_test ${SOURCE_FILES})
target_link_libraries(enn_gpu_op_CLNormalization_test ${LIBRARY_FILES})
add_test(NAME CLNormalization_test COMMAND enn_gpu_op_CLNormalization_test)