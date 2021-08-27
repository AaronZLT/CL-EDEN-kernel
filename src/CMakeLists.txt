cmake_minimum_required(VERSION 3.20)
project(src)

set(SRC_TOP ${CMAKE_SOURCE_DIR})
include(${CMAKE_CURRENT_SOURCE_DIR}/x86_build.cmake)

add_subdirectory(client)
add_subdirectory(common)
add_subdirectory(model)
add_subdirectory(runtime)
add_subdirectory(userdriver/cpu)
add_subdirectory(userdriver/cpu/op_test)  # disabled until user driver mock-up made
#add_subdirectory()

# TODO(hj2020.kim):
# - Use /tmp/ for test_data and /data/vendor/eden/ for Android
# - Check test file IO error handling
