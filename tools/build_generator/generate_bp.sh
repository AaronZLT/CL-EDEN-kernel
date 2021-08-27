#! /bin/bash
# TODO(hoon98.choi, TBD): Change bash shell to python or similar
#set -e -o pipefail

### Directory
top_dir="$(pwd)/../"  # assume that this script runs from top/build/build_tmp/
echo " # Top dir: ${top_dir}"

android_ver="P"  # default
src_top_dir=${top_dir}
cmake_base_dir=${src_top_dir}
output_tmp_dir=${cmake_base_dir}/build_tmp
cmake_define=".."
cmake_cmd_platform="cmake"
verbose_mode=false
enn_profiler_enable=false
frequency_dump_enable=false
utilization_dump_enable=false
sanitizer_enable=false
coverage_enable=false
veloce=false

source ${cmake_base_dir}/machine_info.inc
#echo "${cmake_base_dir}/machine_info.inc"
android_ver=${ANDROID_VER}


### GetOpt
while getopts "t:pfuhrvleams:V:C:" flag; do
  case ${flag} in
    t) product_name="${OPTARG}";;
    r) release_build=true ;;
    v) verbose_mode=true ;;
    h) default_usage; exit 2;;
    e) veloce=true;;
    p) enn_profiler_enable=true;;
    f) frequency_dump_enable=true;;
    u) utilization_dump_enable=true;;
    a) sanitizer_enable=true;;
    m) coverage_enable=true;;
    s) schema_version=${OPTARG};;
    V) version_info=${OPTARG};;
    C) commit_info=${OPTARG};;
    *) "Illegal option."; default_usage; exit -1;;
  esac
done

###### BUILD Utils
function default_usage() {
  echo
  echo " # usage: `basename $0` [options]"
  echo
  echo " [flag]  [option name]   [description]"
  echo " -r      release build"
  echo " -v      verbose build"
  echo " -h      show this"
  echo " -p      enn profiler enable"
  echo " -a      sanitizer enable"
  echo " -s      schema version"
  echo " -e      veloce emulator"
  echo " -t      set framework_top"
  echo -n " -p      product_name    "
    for (( i = 0 ; i < ${#n_target_lists[@]}; i++ )) ; do
	    bd_name=${n_target_lists[$i]}
	    echo -n "${!bd_name[0]}  "
    done
  echo ""
  echo
  echo
  exit 0
}

function printg() {
  echo -e "\033[36m$1\033[0m"
}

function printr() {
  echo -e "\033[31m$1\033[0m"
}

init_product_info ${product_name}

echo " # bp generator configuration: product_name(${product_name}), release_build(${release_build}), verbose(${verbose_mode})"
android_ver=${MC_BUILD_PLATFORM_VER}

if [[ ${android_ver} != "P" && ${android_ver} != "Q" && ${android_ver} != "R" && ${android_ver} != "S" ]]; then
  printg "Android ver [${android_ver}] is not valid"
  exit -1
fi

if [[ ${verified} != "true" ]]; then
  printr "Verification failed. please check machine_info.inc"
  exit -1
fi

if [[ ${verbose_mode} == "true" ]];
then
  BUILD_VERBOSE=true
  cmake_define="${cmake_define} -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"
fi

if [[ ${sanitizer_enable} == "true" ]];
then
  cmake_define="${cmake_define} -DSANITIZER_ENABLE=true"
fi

if [[ ${coverage_enable} == "true" ]];
then
  cmake_define="${cmake_define} -DCOVERAGE_ENABLE=true"
fi

if [[ ${enn_profiler_enable} == "true" ]];
then
  cmake_define="${cmake_define} -DENN_PROFILER_ENABLE=true"
fi

if [[ ${frequency_dump_enable} == "true" ]];
then
  cmake_define="${cmake_define} -DFREQUENCY_DUMP_ENABLE=true"
fi

if [[ ${utilization_dump_enable} == "true" ]];
then
  cmake_define="${cmake_define} -DUTILIZATION_DUMP_ENABLE=true"
fi

if [[ ${release_build} == "true" ]];
then
  cmake_define="${cmake_define} -DRELEASE_BUILD=true"
fi

if [[ ${veloce} == "true" ]];
then
  cmake_define="${cmake_define} -DVELOCE_SOC=true"
fi

echo " # Make header from types.hal... "
cd ../../../tools/hal_converter/;./types_to_header.py
if [ $? -ne 0 ]; then
  printr " # Failed to generate types.hal.h"
  exit -1
fi
cd -

if [[ ${schema_version} != "" ]];
then
  cmake_define="${cmake_define} -DSCHEMA_VERSION=${schema_version}"
fi

cmake_define="${cmake_define} -DBUILD_TARGET=${MC_BUILD_NAME}"
cmake_define="${cmake_define} -DTOP_DIR=${top_dir}"
cmake_define="${cmake_define} -DANDROID_VER=${android_ver}"
cmake_define="${cmake_define} -DCMAKE_DESTNATION_DIR=${top_dir}"

cmake_define="${cmake_define} -DVERSION=\\\"${version_info}\\\""
cmake_define="${cmake_define} -DCOMMIT=\\\"${commit_info}\\\""

echo " # cmake_define : ${cmake_define}"
echo ""

### perform_build
function generate_bp() {
  printg " ## Generate Android.bp start (${MC_BUILD_NAME})  "
  cd ${output_tmp_dir} #/temp
  ${cmake_cmd_platform} ${cmake_define} ${cmake_base_dir}
  cd - &> /dev/null
  printg " ## Generate Android.bp finished (${MC_BUILD_NAME})  "

  rm ${output_tmp_dir} -rf
}

### BUILD MAIN
generate_bp
exit 0
