cmake_minimum_required(VERSION 3.20)
project(enn_cpu_ud_operator_test)

set(SRC_TOP ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # check if local build
include(${SRC_TOP}/x86_build.cmake)
endif()

include_directories(${SRC_TOP})

set(LIBRARY_FILES
    enn_dbg_utils
    ${GTEST_MAIN_LDFLAGS}
    ${GTEST_LDFLAGS})

set(EXTRA_FILES
    ${SRC_TOP}/userdriver/cpu/operators/neon_impl/BboxUtil_batch_single.cpp
    ${SRC_TOP}/userdriver/cpu/operators/neon_impl/BboxUtil.cpp
    ${SRC_TOP}/userdriver/cpu/operators/neon_impl/DetectionOutput.cpp
    ${SRC_TOP}/userdriver/cpu/operators/neon_impl/NormalizedBbox.cpp
)
if(UNIT_TEST)
    add_definitions(-DUNIT_TEST)
endif()
enable_testing()
#-------------------------------------------------------------------#
# Linux(emulator) build is possible for only operator not used NEON #
#-------------------------------------------------------------------#
set(SOURCE_FILES ArgMax_test.cpp ../operators/ArgMax.cpp)
add_executable(ArgMax_test ${SOURCE_FILES})
target_link_libraries(ArgMax_test ${LIBRARY_FILES})
add_test(NAME ArgMax_test COMMAND ArgMax_test)

set(SOURCE_FILES ArgMin_test.cpp ../operators/ArgMin.cpp)
add_executable(ArgMin_test ${SOURCE_FILES})
target_link_libraries(ArgMin_test ${LIBRARY_FILES})
add_test(NAME ArgMin_test COMMAND ArgMin_test)

set(SOURCE_FILES AsymmDequantization_test.cpp ../operators/AsymmDequantization.cpp)
add_executable(AsymmDequantization_test ${SOURCE_FILES})
target_link_libraries(AsymmDequantization_test ${LIBRARY_FILES})
add_test(NAME AsymmDequantization_test COMMAND AsymmDequantization_test)

set(SOURCE_FILES AsymmQuantization_test.cpp ../operators/AsymmQuantization.cpp)
add_executable(AsymmQuantization_test ${SOURCE_FILES})
target_link_libraries(AsymmQuantization_test ${LIBRARY_FILES})
add_test(NAME AsymmQuantization_test COMMAND AsymmQuantization_test)

set(SOURCE_FILES CFUInverter_test.cpp ../operators/CFUInverter.cpp)
add_executable(CFUInverter_test ${SOURCE_FILES})
target_link_libraries(CFUInverter_test ${LIBRARY_FILES})
add_test(NAME CFUInverter_test COMMAND CFUInverter_test)

set(SOURCE_FILES Concat_test.cpp ../operators/Concat.cpp)
add_executable(Concat_test ${SOURCE_FILES})
target_link_libraries(Concat_test ${LIBRARY_FILES})
add_test(NAME Concat_test COMMAND Concat_test)

set(SOURCE_FILES Detection_test.cpp ../operators/Detection.cpp ${EXTRA_FILES})
add_executable(Detection_test ${SOURCE_FILES})
target_link_libraries(Detection_test ${LIBRARY_FILES})
add_test(NAME Detection_test COMMAND Detection_test)

set(SOURCE_FILES Detection_batch_single_test.cpp ../operators/Detection_batch_single.cpp ${EXTRA_FILES})
add_executable(Detection_batch_single_test ${SOURCE_FILES})
target_link_libraries(Detection_batch_single_test ${LIBRARY_FILES})
add_test(NAME Detection_batch_single_test COMMAND Detection_batch_single_test)

set(SOURCE_FILES Flatten_test.cpp ../operators/Flatten.cpp ${EXTRA_FILES})
add_executable(Flatten_test ${SOURCE_FILES})
target_link_libraries(Flatten_test ${LIBRARY_FILES})
add_test(NAME Flatten_test COMMAND Flatten_test)

set(SOURCE_FILES NEONCFUConverter_test.cpp ../operators/NEONCFUConverter.cpp)
add_executable(NEONCFUConverter_test ${SOURCE_FILES})
target_link_libraries(NEONCFUConverter_test ${LIBRARY_FILES})
add_test(NAME NEONCFUConverter_test COMMAND NEONCFUConverter_test)

set(SOURCE_FILES NEONDequantization_test.cpp ../operators/NEONDequantization.cpp)
add_executable(NEONDequantization_test ${SOURCE_FILES})
target_link_libraries(NEONDequantization_test ${LIBRARY_FILES})
add_test(NAME NEONDequantization_test COMMAND NEONDequantization_test)

set(SOURCE_FILES NEONNormalQuantization_test.cpp ../operators/NEONNormalQuantization.cpp)
add_executable(NEONNormalQuantization_test ${SOURCE_FILES})
target_link_libraries(NEONNormalQuantization_test ${LIBRARY_FILES})
add_test(NAME NEONNormalQuantization_test COMMAND NEONNormalQuantization_test)

set(SOURCE_FILES NEONSigDet_test.cpp ../operators/NEONSigDet.cpp ${EXTRA_FILES})
add_executable(NEONSigDet_test ${SOURCE_FILES})
target_link_libraries(NEONSigDet_test ${LIBRARY_FILES})
add_test(NAME NEONSigDet_test COMMAND NEONSigDet_test)

set(SOURCE_FILES Normalization_test.cpp ../operators/Normalization.cpp)
add_executable(Normalization_test ${SOURCE_FILES})
target_link_libraries(Normalization_test ${LIBRARY_FILES})
add_test(NAME Normalization_test COMMAND Normalization_test)

set(SOURCE_FILES Pad_test.cpp ../operators/Pad.cpp)
add_executable(Pad_test ${SOURCE_FILES})
target_link_libraries(Pad_test ${LIBRARY_FILES})
add_test(NAME Pad_test COMMAND Pad_test)

set(SOURCE_FILES Quantization_test.cpp ../operators/Quantization.cpp)
add_executable(Quantization_test ${SOURCE_FILES})
target_link_libraries(Quantization_test ${LIBRARY_FILES})
add_test(NAME Quantization_test COMMAND Quantization_test)

set(SOURCE_FILES Softmax_test.cpp ../operators/Softmax.cpp)
add_executable(Softmax_test ${SOURCE_FILES})
target_link_libraries(Softmax_test ${LIBRARY_FILES})
add_test(NAME Softmax_test COMMAND Softmax_test)

