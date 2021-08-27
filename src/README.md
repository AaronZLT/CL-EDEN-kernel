---
Title: Unified Exynos Neural Network Framework
Output: pdf_document
Date: 2021. 4. 8

---
# Unified Exynos Neural Network Framework

Exynos Neural Network (ENN) framework unifies the existing Exynos Deep Neural Network (EDEN) framework and DSP RT Framework (OFI) by focusing on rearchitecting EDEN framework. EDEN runs NN model’s inference mainly in NPU devices. OFI is mostly used to computer vision application (CV) running on DSP devices.

EDEN and OFI frameworks are basically heterogeneous computing frameworks that support various computing devices: CPU, GPU, NPU, and DSP. Tasks are represented as a directed acyclic graphical (DAG) form in both frameworks: vertices for operations and edges for input/output. Due to those similarity, ENN will unify both frameworks to reduce the cost of software development and maintenance in half. Furthermore, this framework unification will reform the framework implementation by absorbing each framework developers’ experience of development and maintenance.

By supporting both EDEN and OFI interfaces, ENN supports Exynos Neural Network (NN) and Computer Vision (ECV) applications.

- Mobile Neural Network Inference
    - This S/W should support various NN inference. Such as face detection in the camera app.
    - Support all the existing EDEN framework features.
- Mobile Computer Vision via DSP hardware
    - This S/W should support various CV model running on DSP.
    - Support all the existing OFI framework features.

EDEN and OFI frameworks are basically heterogeneous computing frameworks that support various computing devices: CPU, GPU, NPU, and DSP. Tasks are represented as a directed acyclic graphical (DAG) form in both frameworks: vertices for operations and edges for input/output. Due to those similarity, ENN will unify both frameworks to reduce the cost of software development and maintenance in half. Furthermore, this framework unification will reform the framework implementation by absorbing each framework developers’ experience of development and maintenance.

## Code Synchronize & Build Guide
### 1. Code sync
> Currently, we use s-r-mcd-dev platform branch for new framework development
> If you are Android-S based platform, we suggest to use '*universal2100s*' for build parameter.

> new ENN code is built in Android using Android-related libraries
> The build operation includes HIDL libraries as well as Framework source codes.

#### 1-1. Sync platform code
> Required over 80GB space and you can get any version of Android R
> After entering repo sync, you can go outside for a while.
```bash{auto}
## Make directory to sync repo
$ cd { wherever you want to install platform code }
$ mkdir repo; cd repo

## Download Platform Repo // for example
$ export GERRIT_ID={YOUR_GERRIT_ID of 10.166.101.43}
$ mkdir platform; cd platform;
$ repo init -u ssh://${GERRIT_ID}@10.166.101.43:29418/platform/exynos-manifest
  .git -b s-r-mcd-dev --repo-url=ssh://${GERRIT_ID}@10.166.101.43:29418/tools/
  repo.git
$ repo sync -j4
$ cd ..

## Download Product Repo
$ mkdir -r product/dev/universal2100r; cd product/dev/universal2100r;
$ repo init -u ssh://${GERRIT_ID}@10.166.101.43:29418/platform/exynos-manifest
  .git -b product/dev/universal2100_r --repo-url=ssh://${GERRIT_ID}@10.166.101
  .43:29418/tools/repo.git
$ repo sync -j4

## Create Product Symbolic Link in the platform (example)
$ ln -s ~/repo/product/dev/universal2100r ~/repo/platform/product
```
<br>
#### 1-2. Set ANDROID_TOP where you download the platform code

```bash
# Set once
$ export ANDROID_TOP={the starting path of platform}
$ ./build.sh -t universal2100

# Set your shell
$ echo "export ANDROID_TOP={...}" >> ~/.bashrc
$ source ~/.bashrc

# For example
$ export ANDROID_TOP=~/ENN_FRAMEWORK/platform
$ ./build.sh -t universal2100
```

<br>
#### 1-3. sync code of Unified ENN Framework

```bash
$ cd $ANDROID_TOP
$ cd vendor/samsung_slsi/exynos
$ mkdir enn
$ cd enn
$ git clone ssh://${GERRIT_ID}@10.166.101.43:29418/platform/vendor/
  samsung_slsi/exynos/enn/source && scp -p -P 29418 ${GERRIT_ID}@10.166.
  101.43:hooks/commit-msg source/.git/hooks/
```

> Default branch name is **develop**


<br>

### 2. Build & Download Outputs To Your Device
#### 2-1. Build
```bash
$ cd src/
$ ./build.sh -t unviersal2100 -l # build with lib mode (no HIDL)
$ ./build.sh -t universal2100s   # For Android S
$ ./build.sh -h                  # show help message
$ ./build.sh -c                  # clean
```

<br>

* You can find the detail options to use build


```bash
$ ./build.sh -h
==================================================
  Help
==================================================
  -c: Clean directory
  -v: Verbose build
  -l: Library build
  -r: Release build
  -t: target name:
    - universal2100
    - universal2100s
  -g: GENERATE_BP_ONLY
  -a: Enable AddressSanitizer
  -h: show help
==================================================

```

<br>

#### 2-2. Download

> You can use src/push.sh


```bash
$ cd src/
$ ./push.sh -t universal2100
 ### Configurations

 # ANDROID_TOP              : /home/hara/working/s-r-mcd-dev/
 # TARGET                   : universal2100
 # PUSH LIB/BIN ORIGIN VENDOR PATH : /home/hara/working/s-r-mcd-dev/
   /out/target/product/universal2100/vendor
 # PUSH LIB/BIN ORIGIN SYSTEM PATH : /home/hara/working/s-r-mcd-dev/
   /out/target/product/universal2100/system


remount succeeded
 ## download ofi vendors
 -     UPDATED: adb push /home/hara/working/s-r-mcd-dev//out/target/product/
 universal2100/vendor/lib64/vendor.samsung_slsi.hardware.enn@1.0.so /vendor/
 lib64/vendor.samsung_slsi.hardware.enn@1.0.so
 -     UPDATED: adb push /home/hara/working/s-r-mcd-dev//out/target/product/
 universal2100/vendor/lib64/libenn_common_utils.so /vendor/lib64/libenn_comm
 on_utils.so
 -     UPDATED: adb push /home/hara/working/s-r-mcd-dev//out/target/product/
 universal2100/vendor/lib64/libenn_user.so /vendor/lib64/libenn_user.so
 :
 :
 :

```

> *1. If you could see "NOT UPDATED", please check your environment (path,..)*
>
> *2. If this is your first execution, please reboot your device*
>
> *The build output will be generated in `${ANDOIRD_TOP}/out/system` or `${ANDOIRD_TOP}/out/vendor`.*


<br>

### 3. Test (internal)

* Basically, the code build test executable, and you can find at /vendor/bin/
```bash
adb shell "enn_test_internal"   # You can also see logs in logcat
```

* If you build with HIDL mode (not used *-l*), you have to execute service executable in another window(session)
```bash
$ adb shell "/vendor/bin/hw/vendor.samsung_slsi.hardware.enn@1.0-service"
```

* run x86 unit tests and their test coverages
```bash
src $ rm -r build coverage  # remove src/build and src/coverage directory if exist
src $ python3 coverage.py  # build in src/build, run unit tests by ctest, and generate coverage report in src/coverage
```

<br>
### References
* *Public header files: {TOP}/src/client/enn_api-public.h*
* *API usage references: {top}/src/cilent/enn_api_test.cc  {keyword: ENN_GT_API_APPLICATION_TEST}*

### 4. Simple Test Example
* There are two steps to run an IV3 model.
#### 4-1. Run a service process.
* It would be better to set the CPU affinity of this thread into CPU middle cluster consisting of core 4, 5, 6 by using the taskset tool.
```bash
adb shell "taskset 70 /vendor/bin/hw/vendor.samsung_slsi.hardware.enn@1.0-service"
```

#### 4-2. Run a test
* If you want to run an IV3 model 100 times, we could do that by using environment variable called ENN_ITER as follows.
```bash
adb shell "export ENN_ITER=100"
```
* Run an IV3 model with setting the CPU affinity of this thread into the CPU middle cluster.
```bash
adb shell "taskset 70 enn_test_internal --gtest_filter=ENN_GT_API_APPLICATION_TEST.model_test_pamir_npu_inception_v3_golden_match"
```
