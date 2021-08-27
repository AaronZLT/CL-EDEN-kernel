# GPU Userdriver operator

## Usage of 'gpu_op_test.py' script
- Help
    ```bash
    $ python3 gpu_op_test.py -h

    usage: gpu_op_test.py [-h] [--command COMMAND] [--target TARGET] [operator [operator ...]]

    ** GPU userdriver operator UnitTest Script **

    positional arguments:
    operator              select operator ['normalization',] # It can be changed by implementing tests

    optional arguments:
    -h, --help            show this help message and exit
    --command COMMAND, -c COMMAND
                            select command dict_keys(['build', 'push', 'run', 'lcov'])  # default: run
    --target TARGET, -t TARGET
                            select target dict_keys(['emulator', 'all'])      # default: universal2100
    ```
&nbsp;
- How to build operator unittest
    - Use command option '-c build'
    - Target option '-t universal2100'(default) is omitable.
    - Can build for 'all' or with variety operator names.
    ```bash
    $ python3 gpu_op_test.py -t [universal2100|emulator] -c build [all|normalization ...]
    ```
&nbsp;
- How to push operator unittest executable
    - Use command option '-c push'
    - Target option '-t universal2100'(default) is omitable.
    - Can push libray and test executable for 'all' or with variety operator names.
    - Push command is necessary for only target device, but not for emulator.
    ```bash
    $ python3 gpu_op_test.py -t universal2100 -c push [all|normalization ...]
    ```
&nbsp;
- How to test operator unittest
    - Use command option '-c run'
    - Target option '-t universal2100'(default) is omitable.
    - Can execute unittest for 'all' or with variety operator names.
    - In case of emulator target, 'ctest' will be executed with built test executables previously.
    ```bash
    $ python3 gpu_op_test.py -t [universal2100|emulator] -c run [all|normalization ...]
    ```
&nbsp;
- How to create coverage report
    - Use command option '-c lcov'
    - Now, only emulator target is supported. But not device target yet.
    - Can execute all unittest(to create .gcda file) and lcov command. Then, generate html report with lcov result(.info file)
    - All report files(and directory) will be in 'coverage' directory.
    ```bash
    $ python3 gpu_op_test.py -t emulator -c lcov
    ```
    - Coverage results
        ```bash
        amber.png
        common/
        cpu/
        emerald.png
        gcov.css
        glass.png
        index-sort-f.html
        index-sort-l.html
        index.html  # Web page file of coverage report
        ruby.png
        snow.png
        updown.png
        usr/
        ```
&nbsp;
## Directory structure of Userdriver operator
```bash

source/src/userdriver

├── gpu
│   ├── CMakeLists.txt
│   ├── common
│   │   ├── CLBuffer.hpp
│   │   ├── CLComputeLibrary.cpp
│   │   ├── CLComputeLibrary.hpp
│   │   ├── CLIncludes.hpp
│   │   ├── CLKernels.hpp
│   │   ├── CLOperators.hpp
│   │   ├── CLParameter.hpp
│   │   ├── CLPlatform.cpp
│   │   ├── CLPlatform.hpp
│   │   ├── CLRuntime.cpp
│   │   ├── CLRuntime.hpp
│   │   ├── CLTensor.cpp
│   │   └── CLTensor.hpp
│   ├── gpu_op_constructor.cc
│   ├── gpu_op_constructor.h
│   ├── gpu_op_executor.cc
│   ├── gpu_op_executor.h
│   ├── gpu_userdriver.cc
│   ├── gpu_userdriver.h
│   ├── gpu_userdriver_test.cc
│   ├── operators
│   │   ├── cl_kernels
│   │   ├── CLNormalization.cpp
│   │   └── CLNormalization.hpp
│   ├── op_test
│   │   ├── Android.bp
│   │   ├── CMakeLists.base
│   │   ├── CMakeLists.txt
│   │   ├── gpu_op_normalization_test.cpp
│   │   ├── gpu_op_test.py
│   │   ├── normalization_test.cpp
│   │   ├── Readme.md
│   │   └── test_function.hpp
│   └── third_party
│       ├── half
│       └── opencl_stub
