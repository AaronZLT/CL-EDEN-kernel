## ENN_SAMPLE (external)

1. How to build with NDK
    ```bash
    $ cd jni/
    $ ./build_ndk.sh

    OUTPUT is in lib/arm64-v8a/enn_sample_external
    ```

2. How to build with Android platform
    ```bash
    $ cd jni/
    $ ./build_mm.sh

    OUTPUT is in ${ANDROID_TOP}/out/target/product/${TARGET}/vendor/bin/enn_sample_external
    ```

3. Push sample models and then run test executable
    ```bash
    $ ./push_and_test.sh
    ```

4. Test result
    ```bash
    [enn_sample_1]
      # Filename: /data/vendor/enn/models/pamir/NPU_IV3/NPU_InceptionV3.nnc
      # Input: 1, Output: 1
    [ 0] IFM: va: 0x709317c000, size:  268203
    [ 0] OFM: va: 0x70985d7000, size:    4000
        [SUCCESS] Result is matched with golden out

    [enn_sample_2]
      # Filename: /data/vendor/enn/models/pamir/NPU_IV3/NPU_InceptionV3.nnc
      # Input: 1, Output: 1
    [ 0] IFM: va: 0x709317c000, size:  268203
    [ 0] OFM: va: 0x70985d7000, size:    4000
        [SUCCESS] Result is matched with golden out

    [enn_sample_3]
    Session num : 2
      # Filename: /data/vendor/enn/models/pamir/NPU_IV3/NPU_InceptionV3.nnc
    Session[0] allocate & load
      # Input: 1, Output: 1
    [ 0] IFM: va: 0x709317c000, size:  268203
    Session[1] allocate & load
      # Input: 1, Output: 1
    [ 0] IFM: va: 0x709309b000, size:  268203
    Session[0] execute
    Session[1] execute
    Session[0] golden compare & release
    [ 0] OFM: va: 0x70985d7000, size:    4000
        [SUCCESS] Result is matched with golden out
    Session[1] golden compare & release
    [ 0] OFM: va: 0x7097321000, size:    4000
        [SUCCESS] Result is matched with golden out
    ```