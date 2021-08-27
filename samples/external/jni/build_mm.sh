#/bin/bash

if [ -z "$ANDROID_TOP" ]; then
    echo ""
    echo " ## Please set \$ANDROID_TOP as a top of android directory."
        echo " Example)"
    echo "   $ export ANDROID_TOP=<android dir> or add this at ~/.bashrc"
    echo ""
        exit 0;
fi

function disable_platform_bp()
{
  mv_src=$ANDROID_TOP/vendor/samsung_slsi/hardware/enn/1.0/Android.bp
  mv_dst=$ANDROID_TOP/vendor/samsung_slsi/hardware/enn/1.0/Android.bp.tmp
  if [[ -f "${mv_src}" ]]; then
    mv ${mv_src} ${mv_dst}
  fi
  mv_src=$ANDROID_TOP/vendor/samsung_slsi/exynos/enn/Android.bp
  mv_dst=$ANDROID_TOP/vendor/samsung_slsi/exynos/enn/Android.bp.tmp
  if [[ -f "${mv_src}" ]]; then
    mv ${mv_src} ${mv_dst}
  fi
}

disable_platform_bp
mv Android.bp.sample Android.bp
cd $ANDROID_TOP
source build/envsetup.sh
lunch full_erd9925_s-eng
cd $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/samples/external/jni
mm -j16
mv Android.bp Android.bp.sample

echo "DONE!"
