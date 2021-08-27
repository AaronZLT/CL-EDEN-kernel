---
Title: Exynos Neural Network Test Framework
Output: pdf_document
Date: 2021. 7. 13

---
# Exynos Neural Network Test Framework

Exynos Neural Network (ENN) Test Framework provides an easy-to-use test for ENN framework. The ENN Test Framework consists of test app and python scripts. Test app performs end-to-end test with a model file, input files, and golden output files to verify model, ENN and hardware modules. Compared with full-level unit test, ENN test app provides fully-configurable test, able to run various models with various options.
Python scripts support more complicated test cases include SQE test cases, by using test app. It could be used by automated test tool such as Continuous Integration (CI) test.

#### References
 - Confluence page for EnnTest_v2 guide : https://code.sec.samsung.net/confluence/display/NGSC/EnnTest_v2+guide+page
 - Confluence page for test script guide : https://code.sec.samsung.net/confluence/display/NGSC/ENN+test+script+guide+page
<br><br>

## Overview
The ENN Test Framework consists of test app and python scripts.
The following shows the structure of ENN Test Framework.

```bash
test
 戍式式 test_app      # Test app
 戌式式 script   # Python scripts
```

<br><br>

## Build & Test Guide
### 1. How to build test app
#### 1. Use mm build
ENN test framework uses Android mm build to build test app.
```bash
$ cd test/test_app/
$ mm -j8

# after build successed

$ ls {platform_root}/out/target/product/{product_name}/vendor/bin/EnnTest_v2_lib   # Lib test app location
$ ls {platform_root}/out/target/product/{product_name}/vendor/bin/EnnTest_v2_service   # Service test app location
```

#### 2. Use build script
source/src/build.sh also generate ENN test executable.<br/>
To more detail about build.sh, please check readme from 'source/src/README.md'
```bash
$ cd source/src/
$ build.sh -t {target_device}

# after build successed

$ ls {platform_root}/out/target/product/{product_name}/vendor/bin/EnnTest_v2_lib   # Lib test app location
$ ls {platform_root}/out/target/product/{product_name}/vendor/bin/EnnTest_v2_service   # Service test app location
```

<br>

### 2. How to test
#### 2-1. Using test app
```bash
# push EnnTest_v2 (both lib and service test has same process)
$ adb push EnnTest_v2_lib /vendor/bin/
$ adb push {test_vector} /data/vendor/enn/

# run end-to-end test
$ adb shell "EnnTest_v2_lib --model {model_file} --input {input_file} --golden {golden_file}"

# run multiple input / output model
$ adb shell "EnnTest_v2_lib --model {model_file} --input {input_file_1} {input_file_2} --golden {golden_file_1} {golden_file_2}"

# run with full path of test vectors
$ adb shell "EnnTest_v2_lib --model {model_full_path} --input {input_full_path} --golden {golden_full_path}"

# help message shows more options
$ adb shell "EnnTest_v2_lib --help"
```

<br>

#### 2-1. Using python scripts
```bash
# after push EnnTest_v2, and modle files
$ adb push EnnTest_v2_lib /vendor/bin/
$ adb push {test_vector} /data/vendor/enn/

# run all tc
$ python3 enn_test.py --run e2e   # e2e test
$ python3 enn_test.py --run sqe   # sqe test

# run test with tc num
$ python3 enn_test.py --e2e_tc {tc_num}    # e2e test
$ python3 enn_test.py --sqe_tc {tc_num}    # sqe test

# run test with tc list
$ python3 enn_test.py --e2e_tc [1,1,2,4]   # e2e test
$ python3 enn_test.py --sqe_tc [1,3,5]   # sqe test

# help message shows more options
$ python3 enn_test.py --help
```

<br><br>

## Test script config file
ENN test script (enn_test.py) uses yaml as test config files. (source/test/script/test_config/*.yaml)
### 1. How to add new model for e2e test
#### 1. Add new test vector to 'test_vector.yaml'

```bash
$ vi source/test/script/test_config/test_vector.yaml

# test vector
test_vector:
  - ...
  - Model_Key:
        model: model.nnc
        input:
          - input1.bin
          - input2.bin
        golden:
          - golden1.bin
          - golden2.bin
        threshold: *default_npu
```
 - Model_Key : should be unique value (identifier of this test vector)
 - model, input, golden : nnc, input, golden file name
 - threshold : threshold of the model. Default value is *default_npu

<br>

#### 2. Add new test vector to 'e2e_test.yaml'
```bash
$ vi source/test/script/test_config/e2e_test.yaml

# model
models:
  - default_models: &default_models
    - ...
    - Model_Key # Key from 'test_vector.yaml'


# E2E Testcase
e2e:
    - ...
```
 - Add Model_Key (same key in test_vector.yaml) in default_models list