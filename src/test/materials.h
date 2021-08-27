#ifndef SRC_TEST_MATERIALS_H_
#define SRC_TEST_MATERIALS_H_

#include <string>
#include <vector>

#ifdef ENN_ANDROID_BUILD
#define TEST_FILE_PATH "/data/vendor/enn/models/"
#else
#define TEST_FILE_PATH "./test_data/"
#endif  // ENN_ANDROID_BUILD

#define TEST_FILE(M) ((TEST_FILE_PATH + std::string(M)).c_str())

/**
 * Model files for Olympus
 */
namespace OLYMPUS {

namespace NPU {

namespace IV3 {
const std::string NNC = TEST_FILE("olympus/NPU_IV3/NPU_INCEPTIONV3.nnc");
const std::string INPUT = TEST_FILE("olympus/NPU_IV3/NPU_INCEPTIONV3_input_data.bin");
const std::string GOLDEN = TEST_FILE("olympus/NPU_IV3/NPU_INCEPTIONV3_golden_data.bin");

namespace NCP {
const std::string BIN = TEST_FILE("olympus/NPU_IV3/NCP/ncp.bin");
const std::string INPUT = TEST_FILE("olympus/NPU_IV3/NCP/input.bin");
const std::string GOLDEN = TEST_FILE("olympus/NPU_IV3/NCP/output_golden.bin");
}  // namespace NCP

}  // namespace IV3

}  // namespace NPU

}  // namespace OLYMPUS

/**
 * Model files for Pamir
 */
namespace PAMIR {

namespace NPU {

namespace IV3 {
const std::string NNC = TEST_FILE("pamir/NPU_IV3/NPU_InceptionV3.nnc");
const std::string INPUT = TEST_FILE("pamir/NPU_IV3/NPU_InceptionV3_input_data.bin");
const std::string GOLDEN = TEST_FILE("pamir/NPU_IV3/NPU_InceptionV3_golden_data.bin");

namespace NCP {
const std::string BIN = TEST_FILE("pamir/NPU_IV3/NCP/input.bin");
const std::string INPUT = TEST_FILE("pamir/NPU_IV3/NCP/input.bin");
const std::string GOLDEN = TEST_FILE("pamir/NPU_IV3/NCP/output_golden.bin");
}  // namespace NCP

}  // namespace IV3

namespace DLV3 {
const std::string NNC = TEST_FILE("pamir/DLV3/NPU_MV2_Deeplab_V3_plus_MLPerf_tflite.nnc");
const std::string INPUT = TEST_FILE("pamir/DLV3/NPU_MV2_Deeplab_V3_plus_MLPerf_tflite_input_data.bin");
const std::string GOLDEN = TEST_FILE("pamir/DLV3/NPU_MV2_Deeplab_V3_plus_MLPerf_tflite_Q_golden_data.bin");
}  // namespace DLV3

namespace EdgeTPU {
const std::string NNC = TEST_FILE("pamir/EdgeTPU/NPU_mobilenet_edgetpu_tflite.nnc");
const std::string INPUT = TEST_FILE("pamir/EdgeTPU/NPU_mobilenet_edgetpu_tflite_input_data.bin");
const std::string GOLDEN = TEST_FILE("pamir/EdgeTPU/NPU_mobilenet_edgetpu_tflite_golden_data.bin");
}  // namespace EdgeTPU

namespace SSD {
const std::string NNC = TEST_FILE("pamir/Mobiledet_SSD/NPU_Mobiledet_SSD_tflite.nnc");
const std::string INPUT = TEST_FILE("pamir/Mobiledet_SSD/NPU_Mobiledet_SSD_tflite_input_data.bin");
const std::string GOLDEN = TEST_FILE("pamir/Mobiledet_SSD/NPU_Mobiledet_SSD_tflite_golden_data.bin");

namespace LEGACY {
const std::string NNC_O0_HW_CFU = TEST_FILE("pamir/Mobiledet_SSD/Mobiledet_SSD_tflite_O0_hw_cfu.nnc");
const std::string NNC_O0_SW_CFU = TEST_FILE("pamir/Mobiledet_SSD/Mobiledet_SSD_tflite_O0_sw_cfu.nnc");
const std::string NNC_O1_HW_CFU = TEST_FILE("pamir/Mobiledet_SSD/Mobiledet_SSD_tflite_O1_hw_cfu.nnc");
const std::string NNC_O1_SW_CFU = TEST_FILE("pamir/Mobiledet_SSD/Mobiledet_SSD_tflite_O1_sw_cfu.nnc");
}  // namespace LEGACY

}  // namespace SSD

namespace OD {

namespace VGA {
const std::string NNC = TEST_FILE("pamir/ObjectDetect/OD_VGA_RGB3P.nnc");
const std::string INPUT = TEST_FILE("pamir/ObjectDetect/input_VGA.bin");
const std::vector<std::string> GOLDENS = {
    TEST_FILE("pamir/ObjectDetect/out_VGA_13.bin"), TEST_FILE("pamir/ObjectDetect/out_VGA_14.bin"),
    TEST_FILE("pamir/ObjectDetect/out_VGA_15.bin"), TEST_FILE("pamir/ObjectDetect/out_VGA_16.bin"),
    TEST_FILE("pamir/ObjectDetect/out_VGA_17.bin"), TEST_FILE("pamir/ObjectDetect/out_VGA_18.bin"),
    TEST_FILE("pamir/ObjectDetect/out_VGA_19.bin"), TEST_FILE("pamir/ObjectDetect/out_VGA_20.bin"),
    TEST_FILE("pamir/ObjectDetect/out_VGA_21.bin"), TEST_FILE("pamir/ObjectDetect/out_VGA_22.bin"),
    TEST_FILE("pamir/ObjectDetect/out_VGA_23.bin"), TEST_FILE("pamir/ObjectDetect/out_VGA_24.bin"),
};
}  // namespace VGA

namespace QVGA {
const std::string NNC = TEST_FILE("pamir/ObjectDetect/OD_QVGA_RGB3P.nnc");
const std::string INPUT = TEST_FILE("pamir/ObjectDetect/input_QVGA.bin");
const std::vector<std::string> GOLDENS = {
    TEST_FILE("pamir/ObjectDetect/out_QVGA_11.bin"), TEST_FILE("pamir/ObjectDetect/out_QVGA_12.bin"),
    TEST_FILE("pamir/ObjectDetect/out_QVGA_13.bin"), TEST_FILE("pamir/ObjectDetect/out_QVGA_14.bin"),
    TEST_FILE("pamir/ObjectDetect/out_QVGA_15.bin"), TEST_FILE("pamir/ObjectDetect/out_QVGA_16.bin"),
    TEST_FILE("pamir/ObjectDetect/out_QVGA_17.bin"), TEST_FILE("pamir/ObjectDetect/out_QVGA_18.bin"),
    TEST_FILE("pamir/ObjectDetect/out_QVGA_19.bin"), TEST_FILE("pamir/ObjectDetect/out_QVGA_20.bin"),
};
}  // namespace QVGA

}  // namespace OD

}  // namespace NPU

namespace DSP {

namespace IV3 {
const std::string NNC = TEST_FILE("pamir/DSP_IV3/DSP_INCEPTIONV3_PAMIR_EVT0.nnc");
const std::string INPUT = TEST_FILE("pamir/DSP_IV3/DSP_INCEPTIONV3_PAMIR_EVT0_input_data.bin");
const std::string GOLDEN = TEST_FILE("pamir/DSP_IV3/DSP_INCEPTIONV3_PAMIR_EVT0_golden_data.bin");

namespace UCGO {
const std::string BIN = TEST_FILE("pamir/DSP_IV3/ucgo/dsp_iv3_pamir.ucgo");
const std::string INPUT = TEST_FILE("pamir/DSP_IV3/ucgo/conv1_3x3_s2_shuffled_in.bin");
const std::string GOLDEN = TEST_FILE("pamir/DSP_IV3/ucgo/classifier_shuffled_out.bin");
}  // namespace UCGO

}  // namespace IV3

namespace CGO {

const std::string Gaussian3x3 = TEST_FILE("pamir/CGO/gaussian3x3_9925.cgo");
const std::string IV3 = TEST_FILE("pamir/CGO/DSP_INCEPTIONV3.cgo");

}  // namespace CGO

namespace NFD {

namespace VGA {
    const std::string NNC = TEST_FILE("pamir/DSP_NFD/nfd_vga.nnc");
}

namespace QVGA {
    const std::string NNC = TEST_FILE("pamir/DSP_NFD/nfd_qvga.nnc");
}
}  // namespace NFD

}  // namespace DSP

namespace GPU {

namespace IV3 {
const std::string NNC = TEST_FILE("pamir/GPU_IV3/GPU_InceptionV3.nnc");
const std::string INPUT = TEST_FILE("pamir/GPU_IV3/GPU_InceptionV3_input_data.bin");
const std::string GOLDEN = TEST_FILE("pamir/GPU_IV3/GPU_InceptionV3_golden_data.bin");
}  // namespace IV3


namespace MobileBERT {
    const std::string NNC = TEST_FILE("pamir/GPU_MobileBert/GPU_mobile_bert_gpu.nnc");
    const std::vector<std::string> INPUTS = {
        TEST_FILE("GPU_mobile_bert_gpu_input.bin"),
        TEST_FILE("GPU_mobile_bert_gpu_input1.bin"),
        TEST_FILE("GPU_mobile_bert_gpu_input2.bin"),
    };
    const std::vector<std::string> GOLDENS = {
        TEST_FILE("GPU_mobile_bert_gpu_golden.bin"),
        TEST_FILE("GPU_mobile_bert_gpu_golden1.bin"),
    };
}  // namespace MobileBERT

namespace OPERATOR {
const std::string CONV_FLOAT = TEST_FILE("pamir/GPU_operators/conv_float.nnc");
const std::string ADD = TEST_FILE("pamir/GPU_operators/add_v1_2.nnc");
}  // namespace OPERATOR

}  // namespace GPU

}  // namespace PAMIR

#endif  // SRC_TEST_MATERIALS_H_
