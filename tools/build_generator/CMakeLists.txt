# ----------------------------------------------------------------------------
#  Copyright (c) 2020 Samsung Electronics Co., Ltd.
#
#    for ENN Framework
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.4)

include ("${TOP_DIR}/parameters.cmake")

set(ENN_HIDL_VER 1.0)

# HIDL for Android R
set(NAME "${NAME}HIDL Build, Android ${ANDROID_VER}")
set(ANDROID_HIDL_LIBS \"vendor.samsung_slsi.hardware.enn@1.0\",)
set(ANDROID_HIDL_LIBS ${ANDROID_HIDL_LIBS}\"libhidlbase\",)
set(ANDROID_HIDL_LIBS ${ANDROID_HIDL_LIBS}\"libhidlmemory\",)
set(ANDROID_HIDL_LIBS ${ANDROID_HIDL_LIBS}\"android.hidl.memory@1.0\",)

set (ANDROID_HIDL_SO_SHARED_LIBS \"libhidlbase\",)
set (ANDROID_HIDL_SO_SHARED_LIBS ${ANDROID_HIDL_SO_SHARED_LIBS}\"liblog\",)
set (ANDROID_HIDL_SO_SHARED_LIBS ${ANDROID_HIDL_SO_SHARED_LIBS}\"libutils\",)
set (ANDROID_HIDL_SO_SHARED_LIBS ${ANDROID_HIDL_SO_SHARED_LIBS}\"libcutils\",)

if (${ANDROID_VER} STREQUAL "Q")
    set(ANDROID_HIDL_LIBS ${ANDROID_HIDL_LIBS}\"libhidltransport\",)
    set(ANDROID_HIDL_SO_SHARED_LIBS ${ANDROID_HIDL_SO_SHARED_LIBS}\"libhidltransport\",)
    set(ANDROID_HIDL_SO_SHARED_LIBS ${ANDROID_HIDL_SO_SHARED_LIBS}\"libhwbinder\",)
endif()

if (${ANDROID_VER} STREQUAL "S")
    set(MEMORY_ALLOCATOR_LIB  ${MEMORY_ALLOCATOR_LIB}\"libion\",)
    set(MEMORY_ALLOCATOR_CODE ${MEMORY_ALLOCATOR_CODE}\"common/enn_memory_allocator_ion.cc\",)
else ()
    set(MEMORY_ALLOCATOR_LIB ${MEMORY_ALLOCATOR_LIB}\"libdmabufheap\",)
    set(MEMORY_ALLOCATOR_CODE ${MEMORY_ALLOCATOR_CODE}\"common/enn_memory_allocator_dmabufheap.cc\",)
endif()

if (${RELEASE_BUILD})
    set(NAME "${NAME}(Release mode)")
    set(ENN_BUILD_RELEASE true)
    else()
    set(NAME "${NAME}(Dev mode)")
endif()

# Set Memory allocator for Platform
if (${ANDROID_VER} STREQUAL "S")
    set(BUILD_FLAGS_ION \"-DENN_ALLOCATE_DMABUFHEAP\",)
    set(BUILD_FLAGS_DD \"-DENN_UDD\",)
    set(ANDROID_DEVICE_BUFFER_LIBS \"libdmabufheap\",)
else ()
    set(BUILD_FLAGS_ION \"-DENN_ALLOCATE_ION\",)
    set(BUILD_FLAGS_DD \"-DENN_NPU_DSP_DD\",)
    set(ANDROID_DEVICE_BUFFER_LIBS \"libion\",)
endif()

if (${BUILD_TARGET})
    set(BUILD_TARGET ${BUILD_TARGET})
endif()

# set Android.bp file names
set(LIB_FRAMEWORK_BP_SRC_FILE "${CMAKE_SOURCE_DIR}/Android.bp.in")
set(LIB_FRAMEWORK_BP_DST_FILE "${CMAKE_DESTNATION_DIR}/Android.bp.tmp")  # Tmp file
set(LIB_FRAMEWORK_HIDL_BP_SRC_FILE "${CMAKE_SOURCE_DIR}/../../1.0/Android.bp.in")
set(LIB_FRAMEWORK_HIDL_BP_DST_FILE "${CMAKE_SOURCE_DIR}/../../1.0/Android.bp.source")

# Get firmware parser version (with md5sum)
#set(FW_GET_CMD "../get_fw_link_info.sh")
#set(FW_GET_CMD_ARG1 "..")
#execute_process(COMMAND ${FW_GET_CMD} ${FW_GET_CMD_ARG1} ${FW_GET_CMD_ARG2}
#                                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
#                                    OUTPUT_VARIABLE FW_PARSER_VERSION
#                                    OUTPUT_STRIP_TRAILING_WHITESPACE)
#message(STATUS "FW_PARSER_VERSION: ${FW_PARSER_VERSION}")

# Get RT version (from git commit id)
#set(FRAMEWORK_GET_VER "../get_build_info.sh")
#execute_process(COMMAND ${FRAMEWORK_GET_VER}
#                                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
#                                    OUTPUT_VARIABLE FRAMEWORK_VERSION
#                                    OUTPUT_STRIP_TRAILING_WHITESPACE)
#message(STATUS "FRAMEWORK_VERSION: ${FRAMEWORK_VERSION}")

# Initialize Vars
set(BUILD_VELOCE_FLAGS "")

if(${VELOCE_SOC})
    set (BUILD_VELOCE_FLAGS \"-DVELOCE_SOC\",)
    set (BUILD_EXCEPTION_FLAGS \"-fno-exceptions\",)
    string(REPLACE -DENN_NPU_DSP_DD "" ${BUILD_FLAGS_DD_TMP} ${BUILD_FLAGS_DD})
    set(BUILD_FLAGS_DD ${BUILD_FLAGS_DD_TMP})
else()
    set (BUILD_EXCEPTION_FLAGS \"-fexceptions\",)
endif()

set (BUILD_FLAGS ${BUILD_FLAGS}\"-DENN_ANDROID_BUILD\",)
set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-DENN_ANDROID_BUILD\",)

#### C++ version : c++17 as default
set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-std=c++17\",)

# Coverage
if (${COVERAGE_ENABLE})
  set (BUILD_FLAGS_COVERAGE ${BUILD_FLAGS_COVERAGE}\"-coverage\",)
  set (BUILD_FLAGS_COVERAGE ${BUILD_FLAGS_COVERAGE}\"-DCOVERAGE_ENABLED\",)
  set (LD_FLAGS_COVERAGE ${BUILD_FLAGS_COVERAGE}\"-coverage\",)
  set (BOOL_STRING_COVERAGE "true")
else()
  set (BOOL_STRING_COVERAGE "false")
endif()

if (${SANITIZER_ENABLE})
    set (BUILD_FLAGS ${BUILD_FLAGS}\"-fno-omit-frame-pointer\",)
    set (BUILD_FLAGS ${BUILD_FLAGS}\"-O1\",)
    set (BUILD_FLAGS ${BUILD_FLAGS}\"-g\",)
    set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-g\",)
    set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-fno-omit-frame-pointer\",)
    set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-O1\",)
    set (BUILD_FLAGS_SANITIZER ${BUILD_FLAGS_SANITIZER}\"-DASAN_ENABLED\",)
endif()

# Version and commit info
set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-DCOMMIT_INFO=${COMMIT}\",)
set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-DVERSION_INFO=${VERSION}\",)

if (DEFINED ENN_BUILD_OPTION_PRESET_SCI_FILE)
    set (BUILD_FLAGS ${BUILD_FLAGS}\"-DENN_PRESET_SCI_FILE=\\\"${ENN_BUILD_OPTION_PRESET_SCI_FILE}\\\"\",)
endif()

if (${RELEASE_BUILD})
    set (BUILD_FLAGS ${BUILD_FLAGS}\"-DENN_BUILD_RELEASE\",)
    set (BUILD_FLAGS ${BUILD_FLAGS}\"-Wno-unused-parameter\",)
    set (BUILD_FLAGS ${BUILD_FLAGS}\"-Wno-unused-variable\",)
    set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-DENN_BUILD_RELEASE\",)
    set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-O3\",)
endif()

if (DEFINED SCHEMA_VERSION)
    message("-- SCHEMA_VERSION: " ${SCHEMA_VERSION})
    if (${SCHEMA_VERSION} STREQUAL "nnc_v1")
        set (BUILD_FLAGS ${BUILD_FLAGS}\"-DSCHEMA_NNC_V1\",)
        set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-DSCHEMA_NNC_V1\",)
    endif()
endif()

if (${ENN_PROFILER_ENABLE})
    set (BUILD_FLAGS ${BUILD_FLAGS}\"-DEXYNOS_NN_PROFILER\",)
    set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-DEXYNOS_NN_PROFILER\",)
endif()

if (${FREQUENCY_DUMP_ENABLE})
    set (BUILD_FLAGS ${BUILD_FLAGS}\"-DFREQUENCY_DUMP\",)
    set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-DFREQUENCY_DUMP\",)
endif()

if (${UTILIZATION_DUMP_ENABLE})
    set (BUILD_FLAGS ${BUILD_FLAGS}\"-DUTILIZATION_DUMP\",)
    set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-DUTILIZATION_DUMP\",)
endif()

set(IMPORT_PROFILER_LIB "")
if (${ENN_PROFILER_ENABLE})
    set (IMPORT_PROFILER_LIB ${IMPORT_PROFILER_LIB}\"libenn_profiler\",)
endif()

#set (BUILD_FLAGS_COMMON ${BUILD_FLAGS_COMMON}\"-DENN_FRAMEWORK_VERSION=\\\"${FRAMEWORK_VERSION}\\\"\",)

string(TIMESTAMP BUILT_TIME "%Y-%m-%d %H:%M")
show_parameters()

# generate Android.bp
configure_file("${LIB_FRAMEWORK_BP_SRC_FILE}" "${LIB_FRAMEWORK_BP_DST_FILE}" @ONLY)

# generate hidl Android.bp
# configure_file("${LIB_FRAMEWORK_HIDL_BP_SRC_FILE}" "${LIB_FRAMEWORK_HIDL_BP_DST_FILE}" @ONLY)
