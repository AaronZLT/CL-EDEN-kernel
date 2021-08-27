adb shell "mkdir /data/vendor/enn/models -p"
adb push models/. /data/vendor/enn/models/
adb shell "chmod 664 /data/vendor/enn/models/*"
