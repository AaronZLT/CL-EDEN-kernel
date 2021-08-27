#/bin/bash

NDK_TOOLCHAIN=/opt/toolchains/ndk/android-ndk-r23-beta5/

if [ -z "$NDK_TOOLCHAIN" ]; then
    echo ""
    echo " ## Please set \$NDK_TOOLCHAIN as a top of NDK installed directory."
        echo " Example)"
    echo "   $ export NDK_TOOLCHAIN=<ndk dir> or add this at ~/.bashrc"
    echo ""
        exit 0;
fi

mv Android.mk.sample Android.mk
mv Application.mk.sample Application.mk
$NDK_TOOLCHAIN/ndk-build
mv Android.mk Android.mk.sample
mv Application.mk Application.mk.sample

echo "DONE!"
