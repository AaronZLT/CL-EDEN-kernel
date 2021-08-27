# CPU Userdriver operator

## Usage of 'cpu_op_test.py' script
- Help
    ```bash
    $ python3 cpu_op_test.py -h

    usage: cpu_op_test.py [-h] [--command COMMAND] [--target TARGET] [operator [operator ...]]

    ** CPU userdriver operator UnitTest Script **

    positional arguments:
    operator              select operator ['ArgMax', 'ArgMin', 'Normalization', 'AsymmDequantization', 'ProcessAsymmQuantization', 'CFUConverteruantization', 'Quantization'] # It can be changed by implementing tests

    optional arguments:
    -h, --help            show this help message and exit
    --command COMMAND, -c COMMAND
                            select command dict_keys(['build', 'push', 'run', 'lcov'])  # default: run
    --target TARGET, -t TARGET
                            select target dict_keys(['emulator', 'universal2100'])      # default: universal2100
    ```
&nbsp;
- How to build operator unittest
    - Use command option '-c build'
    - Target option '-t universal2100'(default) is omitable.
    - Can build for 'all' or with variety operator names.
    ```bash
    $ python3 cpu_op_test.py -t [universal2100|emulator] -c build [all|ArgMax ArgMin ...]
    ```
&nbsp;
- How to push operator unittest executable
    - Use command option '-c push'
    - Target option '-t universal2100'(default) is omitable.
    - Can push libray and test executable for 'all' or with variety operator names.
    - Push command is necessary for only target device, but not for emulator.
    ```bash
    $ python3 cpu_op_test.py -t universal2100 -c push [all|ArgMax ArgMin ...]
    ```
&nbsp;
- How to test operator unittest
    - Use command option '-c run'
    - Target option '-t universal2100'(default) is omitable.
    - Can execute unittest for 'all' or with variety operator names.
    - In case of emulator target, 'ctest' will be executed with built test executables previously.
    ```bash
    $ python3 cpu_op_test.py -t [universal2100|emulator] -c run [all|ArgMax ArgMin ...]
    ```
&nbsp;
- How to create coverage report
    - Use command option '-c lcov'
    - Now, only emulator target is supported. But not device target yet.
    - Can execute all unittest(to create .gcda file) and lcov command. Then, generate html report with lcov result(.info file)
    - All report files(and directory) will be in 'coverage' directory.
    ```bash
    $ python3 cpu_op_test.py -t emulator -c lcov
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
            ├── Android.bp
            ├── common/
            │   ├── op_test/
            │   └── operator_interfaces/
            │       ├── common/
            │       └── interfaces/
            │           └── operators/
            │               ├── IArgMax.hpp
            │               ├── INormalization.hpp
            │               ├── ICFUConverter.hpp
            │               └── ...
            └── cpu/
                ├── common/
                │   ├── NEONComputeLibrary.cpp
                │   ├── NEONComputeLibrary.h
                │   ├── NEONIncludes.hpp
                │   ├── NEONMathFun.hpp
                │   ├── NEONOperators.hpp
                │   └── NEONTensor.hpp
                ├── op_test/
                │   ├── Android.bp
                │   ├── CMakeLists.txt
                │   ├── ArgMax_test.cpp
                │   ├── Normalization_test.cpp
                │   ├── CFUConverter_test.cpp
                │   ├── ...
                │   ├── Readme.md
                │   ├── cpu_op_test.py
                └── operators/
                    ├── Android.bp
                    ├── ArgMax.cpp
                    └── ArgMax.hpp
                    ├── Normalization.cpp
                    └── Normalization.hpp
                    ├── neon_impl
                    │   ├── BboxUtil.cpp
                    │   ├── BboxUtil.hpp
                    │   ├── BboxUtil_batch_single.cpp
                    │   ├── BboxUtil_batch_single.hpp
                    │   ├── DetectionOutput.cpp
                    │   ├── DetectionOutput.hpp
                    │   ├── NormalizedBbox.cpp
                    │   └── NormalizedBbox.hpp
                    ├── CFUConverter.cpp
                    └── CFUConverter.hpp
                    └── ...
```
