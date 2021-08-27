cmake_minimum_required(VERSION 3.20)
project(raw_model)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

add_library(raw_model SHARED model.cc)
target_include_directories(raw_model PRIVATE ${SRC_TOP})

if(UNIT_TEST)

endif()