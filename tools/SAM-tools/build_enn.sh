#!/bin/bash
export ALLOW_NINJA_ENV=1
export MY_USER_ID=sat_static.sec
pushd $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src
./build.sh -t universal2100
popd
