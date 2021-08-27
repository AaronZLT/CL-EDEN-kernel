---
Title: Build
Output: pdf_document
author: Hoon Choi
Date: 2020. 12. 10
Reference: https://ogom.github.io/draw_uml/plantuml/
---

#### 1. Build script: file structure for Android
```plantUML
top to bottom direction

skinparam actorStyle awesome
skinparam DefaultTextAlignment center
skinparam arrowColor #000000
skinparam rectangleBackgroundColor #CFE2F3
skinparam fileBackgroundColor #CFE2F3
skinparam databaseBackgroundColor #CFE2F3
skinparam defaultFontSize 15

rectangle "machine_info.inc" as machine_info #D9EAD3
rectangle "parameters.cmake" as parameters #D9EAD3
actor "developer" as developer
rectangle "build.sh" as build
rectangle "generate_bp.sh" as bpgen
rectangle "Android.bp.in" as androidbpin #D9EAD3
rectangle "CMakeList.txt" as cmakelist #D9EAD3
rectangle "Android.bp.tmp" as androidbptmp
circle "CMake" as cmake
rectangle "previously generated\n Android.bp" as prevandroidbp
rectangle "Check update" as update #skyblue
rectangle "Previous Android.bp" as prevandroidbp #CFE2F3
rectangle "Update Android.bp" as update_bp
rectangle "Remove Android.bp.tmp" as remove_bp

developer -> build
developer ~right~> machine_info: system defined
developer ~left~> parameters: user defined
machine_info ..> build
androidbpin ..> cmake
parameters ..> build
build --> bpgen : configure with \n userparameter
cmakelist ..> cmake
bpgen -down-> cmake : send data
cmake --> androidbptmp : generate
androidbptmp --> update
prevandroidbp --> update
note left of update: Check if Android.bp should be updated
update --> update_bp : yes
update --> remove_bp : no
```

#### 2. Libraries structure (2021/06/21)
```plantUML
top to bottom direction

skinparam actorStyle awesome
skinparam DefaultTextAlignment center
skinparam arrowColor #000000
skinparam rectangleBackgroundColor #CFE2F3
skinparam fileBackgroundColor #CFE2F3
skinparam databaseBackgroundColor #CFE2F3
skinparam defaultFontSize 15

rectangle "libenn_model" #SKYBLUE
rectangle "libenn_public_api_cpp" #SKYBLUE
rectangle "libenn_user_lib" #SKYBLUE
rectangle "libenn_user_hidl" #SKYBLUE
rectangle "libenn_user_driver_cpu" #SKYBLUE
rectangle "libenn_user_driver_gpu" #SKYBLUE
rectangle "libenn_user_driver_unified" #SKYBLUE
rectangle "libenn_dispatcher" #SKYBLUE
rectangle "libenn_engine" #SKYBLUE
rectangle "libenn_jsoncpp" #SKYBLUE
rectangle "libenn_preset_config" #SKYBLUE
rectangle "vendor.samsung_slsi.hardware.enn@1.0-service" as service #LIGHTBLUE

rectangle "libdmabufheap" #PINK
rectangle "libion" #PINK
rectangle "libOpenCL" #PINK
rectangle "libenn_memory_manager (static)" as emm1 #LIGHTGREEN
rectangle "libenn_memory_manager (static)" as emm2 #LIGHTGREEN

emm1--libdmabufheap
emm1--libion

libenn_public_api_cpp-->libenn_user_lib
libenn_public_api_cpp-->libenn_user_hidl

libenn_user_lib~left~emm1
libenn_user_hidl~left~emm1
libenn_engine~left~emm2

libenn_user_lib-->libenn_engine : Lib mode(function) call
libenn_user_hidl-.>service : HIDL mode(RPC) call

libenn_user_driver_gpu--libOpenCL

libenn_dispatcher--libenn_user_driver_cpu
libenn_dispatcher--libenn_user_driver_gpu
libenn_dispatcher--libenn_user_driver_unified

libenn_engine-->libenn_dispatcher
libenn_engine-->libenn_model

service-->libenn_engine

libenn_preset_config-->libenn_jsoncpp

```


#### *. Libraries structure (init version, 12/15)
```plantUML
top to bottom direction

skinparam actorStyle awesome
skinparam DefaultTextAlignment center
skinparam arrowColor #000000
skinparam rectangleBackgroundColor #CFE2F3
skinparam fileBackgroundColor #CFE2F3
skinparam databaseBackgroundColor #CFE2F3
skinparam defaultFontSize 15

rectangle "libenn_common_utils.so" as common #SKYBLUE
rectangle "libenn_user.so" as user #SKYBLUE
rectangle "enn_test_internal" as test #ORANGE

rectangle "libenn_osal" as osal
rectangle "libenn_gtest_core" as gtest

rectangle "libgtest" as libgtest #WHITE
rectangle "libgtest_main" as libgtestmain #WHITE

enn_debug_utils.cc-->common
osal-->common
enn_api.cc-->user
enn_osal.c-->osal
enn_gtest_core.cc --> gtest
libgtest-->gtest
libgtestmain-->gtest

gtest~~>test
libgtest-->test
libgtestmain-->test
user-->test

common-->test
common-->user
common->gtest
```