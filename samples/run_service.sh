# run service
adb root
adb remount
echo running service...
adb shell "/vendor/bin/hw/vendor.samsung_slsi.hardware.enn@1.0-service"
