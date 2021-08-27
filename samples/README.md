#### This sample tries to execute model and compare golden data.

1. link library for build
    - lib/libenn_public_api_cpp.so // 64bit library for ARM64

2. build in android
    - $ source (ANDROID_TOP)/build/envsetup.sh
    - $ lunch 61  (erd9925-usrdebug)
    - $ mm -j16

3. sample .nnc files (iv3) and golden data files
    - sample_nnc/

4. Successful result

	erd9925:/ # enn_samples
    ```bash
      # Filename: /data/vendor/enn/models/pamir/NPU_IV3/NPU_InceptionV3.nnc
    [ 0] IFM: va: 0x7c14db2000, size:  268203
    [ 1] OFM: va: 0x7c162eb000, size:    4000
        [SUCCESS] Result is matched with golden out

      # Filename: /data/vendor/enn/models/pamir/ObjectDetect/OD_V2.3.7_VGA_01_25_bankers_no_eltwise.nnc
    [ 0] IFM: va: 0x79021e8000, size:  921600
    [ 1] OFM: va: 0x79020ce000, size: 1152000
        [SUCCESS] Result is matched with golden out
    [ 2] OFM: va: 0x7c14d9f000, size:  345600
        [SUCCESS] Result is matched with golden out
    [ 3] OFM: va: 0x7c14ace000, size:  288000
        [SUCCESS] Result is matched with golden out
    [ 4] OFM: va: 0x7c14eea000, size:   86400
        [SUCCESS] Result is matched with golden out
    [ 5] OFM: va: 0x7c14ec3000, size:   72000
        [SUCCESS] Result is matched with golden out
    [ 6] OFM: va: 0x7c14f27000, size:   21600
        [SUCCESS] Result is matched with golden out
    [ 7] OFM: va: 0x7c14f6e000, size:   19200
        [SUCCESS] Result is matched with golden out
    [ 8] OFM: va: 0x7c150db000, size:    5760
        [SUCCESS] Result is matched with golden out
    [ 9] OFM: va: 0x7c15013000, size:    4800
        [SUCCESS] Result is matched with golden out
    [10] OFM: va: 0x7c162eb000, size:    1440
        [SUCCESS] Result is matched with golden out
    [11] OFM: va: 0x7c1514c000, size:    1440
        [SUCCESS] Result is matched with golden out
    [12] OFM: va: 0x7c15012000, size:     432
        [SUCCESS] Result is matched with golden out

      # Filename: /data/vendor/enn/models/pamir/ObjectDetect/OD_V2.1.6_QVGA_01_26_bankers_no_eltwise.nnc
    [ 0] IFM: va: 0x7c14dbb000, size:  230400
    [ 1] OFM: va: 0x79020d2000, size:  384000
        [SUCCESS] Result is matched with golden out
    [ 2] OFM: va: 0x7c14e15000, size:  115200
        [SUCCESS] Result is matched with golden out
    [ 3] OFM: va: 0x7c14ee8000, size:   96000
        [SUCCESS] Result is matched with golden out
    [ 4] OFM: va: 0x7c14f25000, size:   28800
        [SUCCESS] Result is matched with golden out
    [ 5] OFM: va: 0x7c14ece000, size:   25600
        [SUCCESS] Result is matched with golden out
    [ 6] OFM: va: 0x7c150db000, size:    7680
        [SUCCESS] Result is matched with golden out
    [ 7] OFM: va: 0x7c15013000, size:    6400
        [SUCCESS] Result is matched with golden out
    [ 8] OFM: va: 0x7c162eb000, size:    1920
        [SUCCESS] Result is matched with golden out
    [ 9] OFM: va: 0x7c1514c000, size:    1920
        [SUCCESS] Result is matched with golden out
    [10] OFM: va: 0x7c15012000, size:     576
        [SUCCESS] Result is matched with golden out
    ```
