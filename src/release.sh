#!/bin/bash

FW_TOP="$PWD"
TARGET_NAME="universal9925"
MODE_DAEMON=true
MODE_REMOVE=false
COPY=false
COPY_DIR=""
SKIP_MATERIAL=false

source enn_output.filelist

#ANDROID_TOP=/home/schan.kang/src/android12

if [ -z "$ANDROID_TOP" ]; then
    echo ""
    echo " ## Please set \$ANDROID_TOP as a top of android directory."
        echo " Example)"
    echo "   $ export ANDROID_TOP=<android dir> or add this at ~/.bashrc"
    echo ""
        exit 0;
fi

refresh()
{
    PUSH_ORIGIN_VENDOR_PATH="$ANDROID_TOP/out/target/product/$TARGET_NAME/vendor"
    PUSH_ORIGIN_SYSTEM_PATH="$ANDROID_TOP/out/target/product/$TARGET_NAME/system"
}

show_help()
{
    echo ""
    echo " ============================================"
    echo " ### Help message of push.sh"
    echo " ============================================"
    echo " -t <target name>         : Specify target name (default: $TARGET_NAME) "
    echo " -c <directory name>      : Release files backup to <dir> "
    echo " -S, -s                   : Skip to push material files "
    echo " -D                       : Remove so/bin files "
    echo " -H, -h                   : Help "
    echo " ============================================"
    echo ""
    show_config;
    exit 0;
}

show_config()
{
  refresh
    echo ""
    echo " ### Configurations"
    echo ""
    echo " # TARGET_NAME                     : $TARGET_NAME"
    echo " # ANDROID_TOP                     : $ANDROID_TOP"
    echo " # PUSH LIB/BIN ORIGIN VENDOR PATH : $PUSH_ORIGIN_VENDOR_PATH"
    echo " # PUSH LIB/BIN ORIGIN SYSTEM PATH : $PUSH_ORIGIN_SYSTEM_PATH"
    echo ""
}

while getopts "Ddo:t:hHc:Ss" flag; do
    case $flag in
        D) MODE_REMOVE=true;;
        d) MODE_REMOVE=true;;
        o) TARGET_NAME=$OPTARG;;
        t) TARGET_NAME=$OPTARG;;
        c) COPY=true ; COPY_DIR=$OPTARG;;
        S) SKIP_MATERIAL=true;;
        s) SKIP_MATERIAL=true;;
        h) refresh; show_help;;
        H) refresh; show_help;;
    *) show_help; exit -1;;
    esac
done

show_config;

echo ""

function copy_artifacts_to_path()
{
  ANDROID_OUTPUT_TOP=$ANDROID_TOP/out/target/product/$TARGET_NAME
  FRAMEWORK_OUTPUT_TOP=$1
  echo "   - From ) $ANDROID_OUTPUT_TOP"
  echo "   - To   ) $FRAMEWORK_OUTPUT_TOP"

  SRC_PATH=$ANDROID_OUTPUT_TOP/vendor
  DST_PATH=$FRAMEWORK_OUTPUT_TOP/
  DST_SEVICE=$ANDROID_TOP/vendor/samsung_slsi/hardware/enn/1.0/default

  out_cp_lists $SRC_PATH $DST_PATH "lib" ${enn_lib_list_vendor[@]}
  out_cp_lists $SRC_PATH $DST_PATH "lib64" ${enn_lib_list_vendor[@]}

  mkdir -p ${DST_SEVICE}/bin/
  mkdir -p ${DST_SEVICE}/bin/hw
  out_cp_lists $SRC_PATH $DST_SEVICE "bin" ${enn_bin_list_vendor[@]}

  cp -rf $DST_SEVICE/bin/hw/* $DST_SEVICE/

  rm -rf $DST_SEVICE/bin
}

copy_artifacts_to_path $ANDROID_TOP/vendor/samsung_slsi/exynos/enn

#	Copy ENN headers
echo "Copy ENN headers"
SRC=$ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src/client
DST=$ANDROID_TOP/vendor/samsung_slsi/exynos/enn/include
cp -rf $SRC/enn_api-public.hpp	$DST
cp -rf $SRC/enn_api-type.h		$DST

#	Copy HAL
echo "Copy HAL"
SRC=$ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src/medium/hidl/enn/1.0
DST=$ANDROID_TOP/vendor/samsung_slsi/hardware/enn/1.0/
cp -rf $SRC/*.hal $DST
echo "Done"
