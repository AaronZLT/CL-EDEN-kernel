if [ -z "$ANDROID_TOP" ]; then
    echo ""
    echo " ## Please set \$ANDROID_TOP as a top of android directory."
        echo " Example)"
    echo "   $ export ANDROID_TOP=<android dir> or add this at ~/.bashrc"
    echo ""
        exit 0;
fi

TARGET_NAME="erd9925"
PLATFORM_BUILD=false

while getopts "T:t:Pp" flag; do
    case $flag in
        T) TARGET_NAME=$OPTARG;;
        t) TARGET_NAME=$OPTARG;;
        P) PLATFORM_BUILD=true;;
        p) PLATFORM_BUILD=true;;
    esac
done

PUSH_ORIGIN_VENDOR_PATH="$ANDROID_TOP/out/target/product/$TARGET_NAME/vendor"

echo ""
echo " ### Configurations"
echo ""
echo " # TARGET_NAME                     : $TARGET_NAME"
echo " # ANDROID_TOP                     : $ANDROID_TOP"
echo " # PUSH LIB/BIN ORIGIN VENDOR PATH : $PUSH_ORIGIN_VENDOR_PATH"
echo ""

if [ ! -d $PUSH_ORIGIN_VENDOR_PATH ]; then
  echo "Invalid target name: ${TARGET_NAME}"
  exit 0;
fi

# push
adb root
adb remount
adb shell "mkdir -p /data/vendor/enn/models/pamir/"
adb push sample_nnc/* /data/vendor/enn/models/pamir/
adb push $PUSH_ORIGIN_VENDOR_PATH/lib /vendor/
adb push $PUSH_ORIGIN_VENDOR_PATH/lib64 /vendor/

if [[ ${PLATFORM_BUILD} == true ]]; then
adb push $PUSH_ORIGIN_VENDOR_PATH/bin/enn_sample_external /vendor/bin/
else
adb push libs/arm64-v8a/enn_sample_external /vendor/bin/
fi

# test
adb shell "enn_sample_external"
