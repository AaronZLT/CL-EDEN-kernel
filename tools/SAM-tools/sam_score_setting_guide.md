---
Title: SAM tools installing guide for Unified ENN Framework
Output: pdf_document
author: Hoon Choi
Date: 2021. 04. 09
---
# Install Guide for SAM tools in Samsung
### Highlight
```bash
$ cd src/tools/SAM-tools
$ git clone https://github.sec.samsung.net/RS7-Architectural-Refactoring/SAM-Tools

#### Edit path name of sam_cli.cfg and src_exclude.cfg

$ ./SAM-Tools/sam_cli.sh ./sam_cli.cfg
```

<br>
## Code Sync
```bash
$ git clone https://github.sec.samsung.net/RS7-Architectural-Refactoring/SAM-Tools
  or
$ git clone git@github.sec.samsung.net:RS7-Architectural-Refactoring/SAM-Tools.git # (if use ssh)
```

### Prepare to build / clean shell
* build_enn.sh
```bash
#!/bin/bash
export ALLOW_NINJA_ENV=1
export MY_USER_ID=sat_static.sec
pushd $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src
./build.sh -t universal2100
popd
```
<br>
* clean_enn.sh
```bash
#!/bin/bash
pushd $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src
./build.sh -c
popd
```
> *Please refer to build_enn.sh / clean_enn.sh*


### Configuration

<br>
* src_exclude.cfg

*Exclude dir or files*

```bash
 $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src/build/
 $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src/external/
 $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src/model/schema/
 $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src/client/enn_model_container_test.cc
 $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src/client/enn_api_test.cc
 $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src/client/enn_api-async_test.cc
 $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src/common/enn_memory_manager_test.cc
 $ANDROID_TOP/vendor/samsung_slsi/exynos/enn/source/src/common/enn_debug_test.cc
  :
```

<br>
* sam_cli.cfg in SAM-tools

*You can modify sam_cli.cfg file refer to the following, or set your own environment.*

```bash
SAM_PRJ_NAME="ENN_FRAMEWORK"
BUILD_CMD="$PWD/build_enn.sh"   # Execute sam in same directory includes shells
CLEAN_CMD="$PWD/clean_enn.sh"
SRC_PATH="$PWD/src"             # path to source/src
SRC_LANG="auto_c_cpp"           # c or cpp
SRC_EXCLUDE="$PWD/SAM-Tools/src_exclude.cfg"    # Excluding list
:
SCRA_VERSION=SVACE_3.0.0
:
USE_EMBEDDED_LIB=TRUE
:
#GBS_BUILD=FALSE

```


### Run

```bash
$ ${SAM-tools directory}/sam_cli.sh sam_cli.cfg
```

> you can see the result in output/html/

----
Link to [sam_cli.cfg](sam_cli.cfg)
Link to [src_exclude.cfg](src_exclude.cfg)
Link to [build_enn.sh](build_enn.sh)
Link to [clean_enn.sh](clean_enn.sh)
