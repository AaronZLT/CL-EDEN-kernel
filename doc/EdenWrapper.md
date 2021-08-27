

# Introduction
```
Now we are going to release brand new ENN framework.
One of big issues of ENN deployment is about guaranteeing backward compatibility.
We were providing EDEN framework for last 3 years.

The customer's applications follows 1-apk policy which is supporting all devices by single APK build.
They don't want to make branching per SoC or platform.

So We provide EDEN API Wrapper.
Client have to update their application once after small code changes.
EDEN API Wrapper consists of legacy EDEN API set, but will be linked to ENN framework.
```

## library dependency
- AS-IS (Olympus)
  - Legacy APK (use Eden)
    - libeden_nn_on_system.so (can be included in APK)
    - libeden_rt_stub.edensdk.samsung.so
    - vendor.samsung_slsi.hardware.eden_runtime@1.0.so
- TO-BE (Pamir)
  - Legacy APK (use ENN)
    - libeden_for_enn.so (can be included in APK)
    - libeden_rt_stub.samsung_slsi.so
    - libenn_user.samsung_slsi.so
    - vendor.samsung_slsi.hardware.enn@1.0.so
  - New APK (use ENN)
    - libenn_public_api_cpp.so (can be included in APK)
    - libenn_user.samsung_slsi.so
    - vendor.samsung_slsi.hardware.enn@1.0.so

## Changes in ENN
- Change library name to use ENN from device manufacturer's app
  - libenn_user.so -> libenn_user.samsung_slsi.so
- Add eden wrapper library to suport backward compatibility
  - libeden_rt_stub.samsung_slsi.so
  - libeden_for_enn.so
- To accessible to apps by listing ENN library in .txt files.
  - src : http://10.166.101.43:81/#/c/684625/9/public.libraries-samsung_slsi.txt
```
    libenn_user.samsung_slsi.so
    libeden_rt_stub.samsung_slsi.so
```

### library name rule
 - silicon vendors (starting from Android 7.0) and device manufacturers (starting from Android 9) 
   may choose to provide additional native libraries accessible to apps by putting them under 
   the respective library folders and explicitly listing them in .txt files.
  - vendor (Silicon vendor)
    - library folder : /vendor/lib, /vendor/lib64
    - .txt : /vendor/etc/public.libraries.txt
  - system (device manufacturer)
    - library folder : /system/lib, /system/lib64
    - .txt : /system/etc/public.libraries-libraries-samsung_slsi.txt
- Native libraries in the system partition that are made public by device manufacturers MUST be named lib*COMPANYNAME.so,
    - e.g., libFoo.awesome.company.so. In other words, libFoo.so without the company name suffix MUST NOT be made public.
    - The COMPANYNAME in the library file name MUST match with the COMPANYNAME in the txt file name in which the library name is listed.

reference : https://source.android.com/devices/tech/config/namespaces_libraries?hl=ko

