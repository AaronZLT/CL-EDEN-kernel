#!/bin/bash

FW_TOP="$PWD"
TARGET_NAME="universal2100"
MODE_DAEMON=true
MODE_REMOVE=false
COPY=false
COPY_DIR=""
SKIP_MATERIAL=false

source enn_output.filelist

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

adb remount
echo ""

adb_cmd_with_lists()
{
    local cmd=$1
    local opt=$2
    local path_from=$3
    local path_to=$4
    local dir=$5
    local list=("$@")

    for (( i = 0 ; i < ${#list[@]}; i++ )) ; do
        if [ $i -gt 4 ]; then
            if [ "$cmd" == "download" ]; then
                command="adb $opt $path_from/$dir/${list[$i]} /$path_to/$dir/${list[$i]}"
            elif [ "$cmd" == "rm" ]; then
                command="adb $opt /$path_to/$dir/${list[$i]}"
            else
                echo " ## Not implemented yet: $cmd"
                exit -1;
            fi
            #echo $command
            $command > /dev/null
            if [ "$?" != "0" ]; then
                echo " - NOT UPDATED: $command "
            else
                echo " -     UPDATED: $command "
            fi
        fi
    done
    # $cmd
}

cp_cmd_with_lists()
{
    local cmd=$1
    local opt=$2
    local path_from=$3
    local path_to=$4
    local dir=$5
    local list=("$@")

    for (( i = 0 ; i < ${#list[@]}; i++ )) ; do
        if [ $i -gt 4 ]; then
            if [ "$cmd" == "download" ]; then
                command="cp $path_from/$dir/${list[$i]} $path_to/$dir/${list[$i]}"
            else
                echo " ## Not implemented yet: $cmd"
                exit -1;
            fi
            #echo $command
            $command > /dev/null
            if [ "$?" != "0" ]; then
                echo " - NOT UPDATED: $command "
            else
                echo " -     UPDATED: $command "
            fi
        fi
    done
    # $cmd
}

# Remove lib / bins
if [[ $MODE_REMOVE == true ]]; then
    echo " ## Remove enn vendors "
    adb_rm_lists "" "vendor" "lib64" "${enn_lib_list_vendor[@]}"
    adb_rm_lists "" "vendor" "lib"   "${enn_lib_list_vendor[@]}"
    adb_rm_lists "" "vendor" "bin"   "${enn_bin_list_vendor[@]}"

    echo " ## Remove enn systems "
    adb_rm_lists "" "system" "lib64" "${enn_lib_list_system[@]}"
    adb_rm_lists "" "system" "lib"   "${enn_lib_list_system[@]}"
    adb_rm_lists "" "system" "bin"   "${enn_bin_list_system[@]}"

    echo " ## Remove HIDL - etc "
    adb_rm_lists "" "vendor" "etc/init" "${enn_etc_list[@]}"
fi

if [[ $COPY == false ]]; then
    echo " ## Download enn vendors "
    adb_download_lists "$PUSH_ORIGIN_VENDOR_PATH" "vendor" "lib64" "${enn_lib_list_vendor[@]}"
    adb_download_lists "$PUSH_ORIGIN_VENDOR_PATH" "vendor" "lib"   "${enn_lib_list_vendor[@]}"
    adb_download_lists "$PUSH_ORIGIN_VENDOR_PATH" "vendor" "bin"   "${enn_bin_list_vendor[@]}"
    adb_download_lists "$PUSH_ORIGIN_VENDOR_PATH" "vendor" "etc"   "${enn_etc_list[@]}"

    echo " ## Download enn systems "
    adb_download_lists "$PUSH_ORIGIN_SYSTEM_PATH" "system" "lib64" "${enn_lib_list_system[@]}"
    adb_download_lists "$PUSH_ORIGIN_SYSTEM_PATH" "system" "lib"   "${enn_lib_list_system[@]}"
    adb_download_lists "$PUSH_ORIGIN_SYSTEM_PATH" "system" "bin"   "${enn_bin_list_system[@]}"

#    echo " ## download HIDL - etc "
#    adb_download_lists "$PUSH_ORIGIN_VENDOR_PATH" "vendor" "etc/init" "${enn_etc_list[@]}"

    echo " ## Android push finished"
else
  echo " ## Create Dirctory: ${COPY_DIR}"
    echo " ## Copy enn vendors "
    cp_download_lists "$PUSH_ORIGIN_VENDOR_PATH" "vendor" "lib64" "${enn_lib_list_vendor[@]}"
    cp_download_lists "$PUSH_ORIGIN_VENDOR_PATH" "vendor" "lib"   "${enn_lib_list_vendor[@]}"
    cp_download_lists "$PUSH_ORIGIN_VENDOR_PATH" "vendor" "bin"   "${enn_bin_list_vendor[@]}"

    echo " ## Copy enn systems "
    cp_download_lists "$PUSH_ORIGIN_SYSTEM_PATH" "system" "lib64" "${enn_lib_list_system[@]}"
    cp_download_lists "$PUSH_ORIGIN_SYSTEM_PATH" "system" "lib"   "${enn_lib_list_system[@]}"
    cp_download_lists "$PUSH_ORIGIN_SYSTEM_PATH" "system" "bin"   "${enn_bin_list_system[@]}"

#    echo " ## copy HIDL - etc "
#    cp_download_lists "$PUSH_ORIGIN_VENDOR_PATH" "vendor" "etc/init" "${enn_etc_list[@]}"

    echo " ## Android lib copy finished"
fi

# echo " ## Download materials"
if [[ $SKIP_MATERIAL == false ]]; then
    echo " ## Push materials"
    cd ../materials
    ./push_materials.sh
    cd -
fi

# echo " ####################################################################"
# echo " ## CAUTION : This script push both system & vendor executable !!! ##"
# echo " ####################################################################"