# Developer can modify options under
set(ENN_BUILD_OPTION_RELEASE         false)
set(ENN_BUILD_OPTION_PRESET_SCI_FILE "/vendor/firmware/ext_presets.json")

# Internal Test files
set(ENN_TEST_INTERNAL_FILES ${ENN_TEST_INTERNAL_FILES}\"client/enn_api_test.cc\",)
set(ENN_TEST_INTERNAL_FILES ${ENN_TEST_INTERNAL_FILES}\"client/enn_api-async_test.cc\",)

# Module Test files
set(ENN_MODULE_TEST_FILES ${ENN_MODULE_TEST_FILES}\"common/enn_memory_manager_test.cc\",)
set(ENN_MODULE_TEST_FILES ${ENN_MODULE_TEST_FILES}\"common/enn_debug_test.cc\",)
set(ENN_MODULE_TEST_FILES ${ENN_MODULE_TEST_FILES}\"common/enn_utils_test.cc\",)
set(ENN_MODULE_TEST_FILES ${ENN_MODULE_TEST_FILES}\"client/enn_model_container_test.cc\",)
set(ENN_MODULE_TEST_FILES ${ENN_MODULE_TEST_FILES}\"medium/enn_client_manager_test.cc\",)
set(ENN_MODULE_TEST_FILES ${ENN_MODULE_TEST_FILES}\"model/parser/parser_test.cc\",)
set(ENN_MODULE_TEST_FILES ${ENN_MODULE_TEST_FILES}\"preset/preset_config_test.cc\",)
set(ENN_MODULE_TEST_FILES ${ENN_MODULE_TEST_FILES}\"runtime/engine_test.cc\",)

# Userdriver Test files
set(ENN_USERDRIVER_TEST_FILES ${ENN_USERDRIVER_TEST_FILES}\"userdriver/cpu/cpu_userdriver_test.cc\",)
set(ENN_USERDRIVER_TEST_FILES ${ENN_USERDRIVER_TEST_FILES}\"userdriver/gpu/gpu_userdriver_test.cc\",)
set(ENN_USERDRIVER_TEST_FILES ${ENN_USERDRIVER_TEST_FILES}\"userdriver/unified/npu_userdriver_test.cc\",)
set(ENN_USERDRIVER_TEST_FILES ${ENN_USERDRIVER_TEST_FILES}\"userdriver/unified/dsp_userdriver_test.cc\",)

# CPU operator Test files
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/ArgMax_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/ArgMin_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/Detection_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/Normalization_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/AsymmDequantization_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/AsymmQuantization_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/NEONCFUConverter_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/CFUInverter_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/Concat_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/NEONDequantization_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/NEONNormalQuantization_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/Pad_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/Quantization_test.cpp\",)
set(ENN_CPU_OPERATOR_TEST_FILES ${ENN_CPU_OPERATOR_TEST_FILES}\"userdriver/cpu/op_test/Softmax_test.cpp\",)

# This is utilities for build (don't modify) ##############################
macro(show_var var_name)
  set (extra_desc ${ARGN})
  list (LENGTH extra_desc num_extra_args)
  if (${num_extra_args} GREATER 0)
    list(GET extra_desc 0 var_content)
    message(STATUS "${var_name}: ${${var_content}}")
    set(BUILD_OPTIONS "${BUILD_OPTIONS}\n * ${var_name}: ${${var_content}}")
  else ()
    message(STATUS "${var_name}: ${${var_name}}")
    set(BUILD_OPTIONS "${BUILD_OPTIONS}\n * ${var_name}: ${${var_name}}")
  endif ()
endmacro()

macro(show_parameters)
  show_var(ENN_BUILD_OPTION_RELEASE)
  show_var(ENN_BUILD_OPTION_PRESET_SCI_FILE)
  set(OPTION_DESC ${OPTION_DESC} "\n" )
endmacro()
