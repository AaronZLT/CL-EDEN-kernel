#!/bin/bash -e
# set -e -o pipefail

#===========================================================================
# variables
#===========================================================================
FRAMEWORK_TOP="$PWD"
BUILD_TOOL_DIR="../tools/build_generator"
source ${FRAMEWORK_TOP}/$BUILD_TOOL_DIR/machine_info.inc
source enn_output.filelist

# LIB_BUILD=false #Android hidl build
CLEAN_BUILD=false
VERBOSE_BUILD=false
RELEASE_BUILD=false
ENABLE_EXYNOS_NN_PROFILER=false
ENABLE_FREQUENCY_DUMP=false
ENABLE_UTILIZATION_DUMP=false
GENERATE_BP_ONLY=false
GEN_FILE="$BUILD_TOOL_DIR/Android.bp.tmp"
PREV_FILE="Android.bp"
ENABLE_ASAN=false
ENABLE_COVERAGE_MEASURE=false
VELOCE=false
VERSION_INFO="2.0.0"
COMMIT_INFO="NOT_DEFINED"

#===========================================================================
# functions
#===========================================================================
function show_build_option() {
  echo "=================================================="
  echo " # Build options "
  echo "=================================================="
  echo " # CLEAN_BUILD: ${CLEAN_BUILD}"
  echo " # VERBOSE_BUILD: ${CLEAN_BUILD}"
  echo " # RELEASE_BUILD: ${RELEASE_BUILD}"
  echo " # ENABLE_EXYNOS_NN_PROFILER: ${ENABLE_EXYNOS_NN_PROFILER}"
  echo " # ENABLE_FREQUENCY_DUMP: ${ENABLE_FREQUENCY_DUMP}"
  echo " # ENABLE_UTILIZATION_DUMP: ${ENABLE_UTILIZATION_DUMP}"
  echo " # TARGET_NAME: ${TARGET_NAME}"
  echo " # OUTPUT_NAME: ${OUTPUT_NAME}"
  echo " # GENERATE_BP_ONLY: ${GENERATE_BP_ONLY}"
  echo " # ENABLE_ASAN: ${ENABLE_ASAN}"
  echo " # ENABLE_COVERAGE_MEASURE: ${ENABLE_COVERAGE_MEASURE}"
  echo " # VELOCE: ${VELOCE}"
  echo " # VERSION_INFO: ${VERSION_INFO}"
  echo " # COMMIT_INFO: ${COMMIT_INFO}"
}

function generate_and_update_bp() {
  mkdir $BUILD_TOOL_DIR/build_tmp -p > /dev/null
  cd $BUILD_TOOL_DIR/build_tmp

  GEN_BP_OPTION="../generate_bp.sh -t ${TARGET_NAME}"
  GEN_BP_OPTION="${GEN_BP_OPTION} -V ${VERSION_INFO} -C ${COMMIT_INFO}"

  if [[ ${ENABLE_COVERAGE_MEASURE} == true ]]; then
    GEN_BP_OPTION="${GEN_BP_OPTION} -m"
  fi

  if [[ ${ENABLE_ASAN} == true ]]; then
    GEN_BP_OPTION="${GEN_BP_OPTION} -a"
  fi

  if [[ ${LIB_BUILD} == true ]]; then
    echo "Build generates both lib- and hidl-mode outputs"
  #  GEN_BP_OPTION="${GEN_BP_OPTION} -l"
  fi

  if [[ ${RELEASE_BUILD} == true ]]; then
    GEN_BP_OPTION="${GEN_BP_OPTION} -r"
  fi

  if [[ ${TARGET_NAME} == "universal2100" ]]; then
    SCHEMA_VERSION="nnc_v1"
    #Can be removed later when veloce will be built with Android S for erd9925
    if [[ ${VELOCE} == true ]]; then
      SCHEMA_VERSION="nnc_v2"
    fi
  fi

  if [[ ${SCHEMA_VERSION} != "" ]]; then
    GEN_BP_OPTION="${GEN_BP_OPTION} -s ${SCHEMA_VERSION}"
  fi

  if [[ ${ENABLE_EXYNOS_NN_PROFILER} == true ]]; then
    GEN_BP_OPTION="${GEN_BP_OPTION} -p"
  fi

  if [[ ${ENABLE_FREQUENCY_DUMP} == true ]]; then
    GEN_BP_OPTION="${GEN_BP_OPTION} -f"
  fi

  if [[ ${ENABLE_UTILIZATION_DUMP} == true ]]; then
    GEN_BP_OPTION="${GEN_BP_OPTION} -u"
  fi

  if [[ ${VELOCE} == true ]]; then
    GEN_BP_OPTION="${GEN_BP_OPTION} -e"
  fi

  CMD_RUN_ERR_EXIT "${GEN_BP_OPTION}"
  cd - &> /dev/null

  if [[ -e ${PREV_FILE} ]]; then                            # cond 1. prev file existed
    diff ${GEN_FILE} ${PREV_FILE} -f &> /dev/null
    if [ $? -ne 0 ]; then
      echo " # File is updated. Change Android.bp"
      mv ${GEN_FILE} ${PREV_FILE} -f &> /dev/null
    else
      echo " # Generated file is same as before. Not updated."
      rm ${GEN_FILE} -rf
    fi
  else
    echo " # File is not existed. Generate Android.bp "
    mv ${GEN_FILE} ${PREV_FILE} -f &> /dev/null
    echo "mv ${GEN_FILE} ${PREV_FILE} -f &> /dev/null"
  fi

}

function show_help()
{
  echo "=================================================="
  echo "  Help"
  echo "=================================================="
  echo "  -c: Clean directory"
  echo "  -v: Verbose build"
  echo "  -l: Library build (not yet)"
  echo "  -r: Release build"
  echo "  -t: target name:"
  for (( i = 0 ; i < ${#n_target_lists[@]}; i++ )) ; do
    bd_name=${n_target_lists[$i]}
    echo "    - ${!bd_name[0]}  "
  done
  echo "  -p: Enable ExynosNnProfiler "
  echo "  -f: Enable frequency dump "
  echo "  -u: Enable utilization dump "
  echo "  -g: GENERATE_BP_ONLY"
  echo "  -a: Enable AddressSanitizer "
  echo "  -m: Enable test coverage measurement(cov) "
  echo "  -s: Schema version (nnc_v1)"
  echo "  -e: Build for Veloce Emulator"
  echo "  -o: send output to <dir>"
  echo "  -V: set tag information "
  echo "  -h: show help"
  echo "=================================================="
  echo ""
}

function get_git_tag_version()
{
  set +e
  CUR_PATH=$1
  if [ "$CUR_PATH" = "" ]; then
    CUR_PATH=$PWD
  fi
  git -C $CUR_PATH describe --long > /dev/null
  if [ "$?" -ne 0 ]; then
    COMMIT_INFO="NOT_DEFINED"
  else
    COMMIT_INFO=$(git -C $CUR_PATH describe --long | sed -e 's/-/:/g')
  fi
}

function exit_()
{
  echo " # Exit with Success: $1"
  echo ""
  exit 0
}

function rm_and_show_file()
{
  echo " # CMD: rm $1 -rf"
  rm $1 -rf
}

function build_cmake_clean()
{
  echo "=================================================="
  echo " # Clean built-ext files "
  echo "=================================================="

  rm_and_show_file "$BUILD_TOOL_DIR/CMakeFiles"
  rm_and_show_file "$BUILD_TOOL_DIR/build_tmp"
  rm_and_show_file ${GEN_FILE}
}

function build_clean()
{
  rm_and_show_file ${PREV_FILE}
  echo ""
}

function check_android_top()
{
  if [[ ! -d ${ANDROID_TOP} ]]; then
    echo " # Please set ANDROID_TOP. "
    echo "   ex) export ANDROID_TOP=[/*Top directory of Android*/]"
    echo ""
    exit -1
  fi
}

function printg() {
  echo -e "\033[36m$1\033[0m"
}

function printr() {
  echo -e "\033[31m$1\033[0m"
}

function CMD_RUN_ERR_EXIT() {
  printg " # pwd: ${PWD}"
  printg " # COMMAND START: CMD: $1 $2 $3 $4 $5"
  $1 $2 $3 $4 $5
  if [ $? -ne 0 ]; then
    printr " # COMMAND ERR OCCURS: (ret: $?) CMD: $1 $2 $3 $4 $5"
    exit -1
  fi
  printg " # COMMAND SUCCESS: CMD: $1 $2 $3 $4 $5"
}

disable_bp_list=(
  $ANDROID_TOP/vendor/samsung_slsi/hardware/enn/1.0/Android.bp
  $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/Android.bp
  $ANDROID_TOP/vendor/samsung_slsi/exynos/enn_saidl_driver/Android.bp
)

function disable_platform_bp()
{
  printg " # Disable Android.bp in Platform"
  for (( i = 0 ; i < ${#disable_bp_list[@]}; i++ )) ; do
    my_src=${disable_bp_list[$i]}
    my_dst=${disable_bp_list[$i]}.tmp
    if [[ -f "$my_src" ]]; then
      printg " # Disable BP : $my_src -> $my_dst"
      mv $my_src $my_dst
    fi
  done
}

function enable_platform_bp()
{
  printg " # Enable Android.bp in Platform"
  for (( i = 0 ; i < ${#disable_bp_list[@]}; i++ )) ; do
    my_src=${disable_bp_list[$i]}.tmp
    my_dst=${disable_bp_list[$i]}
    if [[ -f "${my_src}" ]]; then
      printg " # Enable BP : $my_src -> $my_dst"
      mv ${my_src} ${my_dst}
    fi
  done
}

function build_set_lunch()
{
  cd $ANDROID_TOP
  CMD_RUN_ERR_EXIT "source build/envsetup.sh" "&> /dev/null"
  cd - &> /dev/null
  if [ ! $MC_LUNCH_NAME ]; then
    echo " # Lunch name error: Please check "
    exit
  fi
  CMD_RUN_ERR_EXIT "lunch $MC_LUNCH_NAME"
}


function clean_build()
{
  build_cmake_clean
  if [[ ${CLEAN_BUILD} == true ]]; then
    build_clean
    exit 0
  fi
}

function pre_build()
{
  printg " # Pre-Build start... > ${TARGET_NAME}"

  init_product_info ${TARGET_NAME}
  if [ ${verified} != "true" ]; then
    printr " # Could not get product info from machine_info.inc. please check."
    exit -1
  fi
  get_git_tag_version $PWD
  show_build_option
  check_android_top
  generate_and_update_bp
  build_set_lunch
  disable_platform_bp
  printg " # Pre-Build finished... "
  printg " # Generate mediums"
}

function build()
{
  if [ ${GENERATE_BP_ONLY} == "true" ]; then
    return
  fi
  printg " # Build start... "

  cd $FRAMEWORK_TOP/../
  if [ ${ENABLE_ASAN} == "true" ]; then
    CMD_RUN_ERR_EXIT "mm SANITIZE_TARGET=address SANITIZE_HOST=address"
  else
    CMD_RUN_ERR_EXIT "mm "
  fi
  cd $FRAMEWORK_TOP

  printg " # Build finished... "
}

function copy_artifacts_to_path()
{
  printg " # Copy artifacts start..."

  ANDROID_OUTPUT_TOP=$ANDROID_TOP/out/target/product/$TARGET_NAME/
  FRAMEWORK_OUTPUT_TOP=$FRAMEWORK_TOP/$1
  echo "   - From ) $ANDROID_OUTPUT_TOP"
  echo "   - To   ) $FRAMEWORK_OUTPUT_TOP"

  echo " # Remove previous output"
  rm -rf $FRAMEWORK_OUTPUT_TOP

  SRC_PATH=$ANDROID_OUTPUT_TOP/vendor
  DST_PATH=$FRAMEWORK_OUTPUT_TOP/vendor

  mkdir -p ${DST_PATH}/lib
  out_cp_lists $SRC_PATH $DST_PATH "lib" ${enn_lib_list_vendor[@]}

  mkdir -p ${DST_PATH}/lib64/
  out_cp_lists $SRC_PATH $DST_PATH "lib64" ${enn_lib_list_vendor[@]}

  mkdir -p ${DST_PATH}/bin/
  mkdir -p ${DST_PATH}/bin/hw
  out_cp_lists $SRC_PATH $DST_PATH "bin" ${enn_bin_list_vendor[@]}

  mkdir -p ${DST_PATH}/etc/vintf/manifest
  out_cp_lists $SRC_PATH $DST_PATH "etc" ${enn_etc_list[@]}

  printg " # Copy artifacts finished..."
}

function post_build()
{
  printg " # Post build start..."

  if [[ $OUTPUT_NAME != "" ]]; then
    copy_artifacts_to_path $OUTPUT_NAME
  fi
  enable_platform_bp
  printg " # Post build finished..."
}

#===========================================================================
# Build options
#===========================================================================
while getopts "cvehrlt:o:gampfus:V:" flag; do
  case $flag in
    c) CLEAN_BUILD=true;;
    v) VERBOSE_BUILD=true;;
    l) LIB_BUILD=true;;  # No effect
    r) RELEASE_BUILD=true;;
    t) TARGET_NAME="${OPTARG}";;
    o) OUTPUT_NAME="${OPTARG}";;
    g) GENERATE_BP_ONLY=true;;
    a) ENABLE_ASAN=true;;
    m) ENABLE_COVERAGE_MEASURE=true;;
    s) SCHEMA_VERSION=${OPTARG};;
    p) ENABLE_EXYNOS_NN_PROFILER=true;;
    f) ENABLE_FREQUENCY_DUMP=true;;
    u) ENABLE_UTILIZATION_DUMP=true;;
    e) VELOCE=true;;
    V) VERSION_INFO="${OPTARG}";;
    h) show_help; exit 0;;
    *) show_help; exit 0;;
  esac
done

clean_build
pre_build
build
post_build

exit_ "Build process completed."
