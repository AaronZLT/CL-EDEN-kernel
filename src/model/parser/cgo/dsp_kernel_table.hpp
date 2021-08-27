#ifndef SRC_MODEL_PARSER_CGO_DSP_KERNEL_TABLE_HPP_
#define SRC_MODEL_PARSER_CGO_DSP_KERNEL_TABLE_HPP_

enum ofi_kernel_cpu_id {
    OFI_KERNELS_CPU_XXX1 = 0,
    OFI_KERNELS_CPU_XXX2 = 1,
    OFI_KERNELS_CPU_QUENT = 2,
    OFI_KERNELS_CPU_DEQUENT = 3,
    OFI_KERNELS_CPU_DEQUENT_SOFTMAX = 4,
    OFI_KERNELS_CPU_DETECTION_OUTPUT = 5,
    OFI_KERNELS_CPU_PREPROCESS_BGR_TO_BGRX = 6,
    OFI_KERNELS_CPU_PREPROCESS_BGR_TO_BGRX_WITH_DIV_1000000 = 7,
    OFI_KERNELS_CPU_PREPROCESS_RGB_TO_BGRX = 8,
    OFI_KERNELS_CPU_PREPROCESS_Y_TO_YXXX_WITH_DIV_1000000 = 9,
    OFI_KERNELS_CPU_PREPROCESS = 11,
    OFI_KERNELS_CPU_PREPROCESS_IDE = 16,
    OFI_KERNELS_CPU_POSTPROCESS_IDE = 14,
    OFI_KERNELS_CPU_CHANNEL_SHUFFLE = 17,
    OFI_KERNELS_CPU_CHANNEL_DESHUFFLE = 18,
    OFI_KERNELS_CPU_CHANNEL_DESHUFFLE_16_to_4 = 19,
    OFI_KERNELS_CPU_PREPROCESS_RGB_TO_BGRD = 20,
    OFI_KERNELS_CPU_MV1_SSD_ANTUTU_POST = 21,
};

inline const char* get_cpu_kernel_names(int e) {
    static const char* names[] = {
        "DSP",
        "buffer_in_to_out",
        "quantization",
        "dequantization",
        "dequantization_softmax",
        "detection output",
        "preprocess_bgr_to_bgrx",
        "divide float scale by 1000000 and preprocess_bgr_to_bgrx",
        "preprocess_rgb_to_bgrx",
        "divide float scale by 1000000 and preprocess to yxxx",
        "NONE",  // 10
        "cpu_preprocess",
        "NONE",  // 12
        "NONE",  // 13
        "postprocess_image_depth_estimation",
        "NONE",  // 15
        "preprocess_image_depth_estimation",
        "channel_shuffle",
        "channel_deshuffle",
        "channel_deshuffle_16_to_4",
        "ofi_custom_cpu_preprocess_rgb_to_bgrd",
        "mv1_ssd",
    };
    return names[e];
}

#endif  // SRC_MODEL_PARSER_CGO_DSP_KERNEL_TABLE_HPP_
