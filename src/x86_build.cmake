# Common CMake Setting
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "-O2 -W -Wall -Wextra -Werror -pthread")  #TODO: remove all warnings
set(CMAKE_CXX_FLAGS "-O0 -W -Wall -Wextra -pthread" CACHE STRING "cmake flags")

if(ENABLE_ASAN)
message("Address Sanitizer is enabled")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
endif()

if(ENABLE_LSAN)
message("Leak Sanitizer is enabled")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fno-omit-frame-pointer -g")
endif()

if(ENABLE_TSAN)
message("Thread Sanitizer is enabled")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -g")
endif()

add_library(enn_dbg_utils   SHARED  ${SRC_TOP}/common/enn_debug.cc ${SRC_TOP}/common/enn_utils.cc)
target_include_directories(enn_dbg_utils PRIVATE ${SRC_TOP})

if(UNIT_TEST)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")  # Add Test Coverage
find_package(PkgConfig)
pkg_search_module(GTEST REQUIRED gtest_main)
enable_testing()  # to run all TC by ctest or ctest --verbose
endif()