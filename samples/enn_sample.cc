#include "enn_api-public.hpp"
#include "enn_sample_utils.hpp"

namespace enn {
namespace api {
extern EnnReturn EnnSetPreferenceTargetLatency(const uint32_t val);  // NOTE(hoon98.choi): this is not public
                                                                     // Only permitted for OD model internally
}
}  // namespace enn

int do_test(const std::string filename, const std::vector<std::string> &golden_inputs,
            const std::vector<std::string> &golden_outputs) {
    if (enn::api::EnnInitialize()) {
        PRINT_AND_RETURN("Initialize Failed");
    }

    // NOTE) If need to set preperence, call preperence APIs before EnnOpenModel()
    // Set TargetLatency. If you want to set the target latency to 20ms, set it as follows.
    uint32_t target_latency = 20000;
    enn::api::EnnSetPreferenceTargetLatency(target_latency);

    // Set PerfMode (Default: ENN_PREF_MODE_BOOST_ON_EXE)
    uint32_t perf_mode = ENN_PREF_MODE_NORMAL;
    enn::api::EnnSetPreferencePerfMode(perf_mode);

    EnnModelId model_id;

    // Open model and get model_id
    if (enn::api::EnnOpenModel(filename.c_str(), &model_id)) {
        PRINT_AND_RETURN("Open Model Failed: %s", filename.c_str());
    } else {
        PRINT("  # Filename: %s", filename.c_str());
    }

    // generate buffer space before calling EnnAllocateAllBuffers()
    if (enn::api::EnnGenerateBufferSpace(model_id)) {
        PRINT_AND_RETURN("Failed to generate buffer space");
    }

    EnnBufferPtr *buffer_set;
    NumberOfBuffersInfo buffers_info;

    // NOTE)
    // allocate all buffers required in opened model (In, Out)
    // session id = 0, not commit yet
    // Expected result:
    //     buffer_set[0]   : input_0
    //     ...
    //     buffer_set[N]   : input_N
    //     buffer_set[N+1] : output_0
    //     ...
    //     buffer_set[END] : output_END
    if (enn::api::EnnAllocateAllBuffers(model_id, &buffer_set, &buffers_info, 0, false)) {
        PRINT_AND_RETURN("Allocate Buffer Error");
    }

    // User is also able to get buffer information of opened model
    // with EnnGetBuffersInfo, EnnGetBufferInfoByIndex, EnnGetBufferInfoByLabel.
    // get allocated buffer information
    // if (enn::api::EnnGetBuffersInfo(&buffers_info, model_id)) {
    //    PRINT_AND_RETURN("Get Buffer information Error");
    // }

    if (enn::sample_utils::check_valid_buffers_info(buffers_info, golden_inputs.size(), golden_outputs.size())) {
        PRINT_AND_RETURN("Invalid Buffer information");
    }

    uint32_t n_in_buf = buffers_info.n_in_buf;
    uint32_t n_out_buf = buffers_info.n_out_buf;

    // Load input files to input buffers
    int in_idx_start = 0;
    int in_idx_end = n_in_buf;
    if (enn::sample_utils::load_input_files(in_idx_start, in_idx_end, golden_inputs, buffer_set)) {
        PRINT_AND_RETURN("Input file load error");
    }

    // send allocated buffers to service core
    if (enn::api::EnnBufferCommit(model_id)) {
        PRINT_AND_RETURN("Failed to commit buffers");
    }

    // Execute model with commeted buffers
    if (enn::api::EnnExecuteModel(model_id)) {
        PRINT_AND_RETURN("Failed to execute model");
    }

    // Load golden output files to reference buffer and compare with output bufferss
    int out_idx_start = n_in_buf;
    int out_idx_end = n_in_buf + n_out_buf;
    if (enn::sample_utils::load_golden_files_and_compare(out_idx_start, out_idx_end, golden_outputs, buffer_set)) {
        PRINT_AND_RETURN("Golden output file load error");
    }

    // Release buffers
    if (enn::api::EnnReleaseBuffers(buffer_set, n_in_buf + n_out_buf)) {
        PRINT_AND_RETURN("Failed to Release buffers");
    }

    // Close model
    if (enn::api::EnnCloseModel(model_id)) {
        PRINT_AND_RETURN("Failed to close model");
    }

    if (enn::api::EnnDeinitialize()) {
        PRINT_AND_RETURN("Deinitialize Failed");
    }

    return 0;
}

int main() {
    const std::string filename =
        TEST_FILE("pamir/NPU_IV3/NPU_InceptionV3.nnc");
    const std::vector<std::string> golden_in = {
        TEST_FILE("pamir/NPU_IV3/NPU_InceptionV3_input_data.bin"),
    };
    const std::vector<std::string> golden_out = {
        TEST_FILE("pamir/NPU_IV3/NPU_InceptionV3_golden_data.bin"),
    };
    do_test(filename, golden_in, golden_out);

    const std::string file_OD1 =
        TEST_FILE("pamir/ObjectDetect/OD_VGA_RGB3P.nnc");
    const std::vector<std::string> golden_in_OD1 = {
        TEST_FILE("pamir/ObjectDetect/input_VGA.bin"),
    };
    const std::vector<std::string> golden_outs_OD1 = {
        TEST_FILE("pamir/ObjectDetect/out_VGA_13.bin"),
        TEST_FILE("pamir/ObjectDetect/out_VGA_14.bin"),
        TEST_FILE("pamir/ObjectDetect/out_VGA_15.bin"),
        TEST_FILE("pamir/ObjectDetect/out_VGA_16.bin"),
        TEST_FILE("pamir/ObjectDetect/out_VGA_17.bin"),
        TEST_FILE("pamir/ObjectDetect/out_VGA_18.bin"),
        TEST_FILE("pamir/ObjectDetect/out_VGA_19.bin"),
        TEST_FILE("pamir/ObjectDetect/out_VGA_20.bin"),
        TEST_FILE("pamir/ObjectDetect/out_VGA_21.bin"),
        TEST_FILE("pamir/ObjectDetect/out_VGA_22.bin"),
        TEST_FILE("pamir/ObjectDetect/out_VGA_23.bin"),
        TEST_FILE("pamir/ObjectDetect/out_VGA_24.bin"),
    };
    do_test(file_OD1, golden_in_OD1, golden_outs_OD1);

    const std::string file_OD2 =
        TEST_FILE("pamir/ObjectDetect/OD_QVGA_RGB3P.nnc");
    const std::vector<std::string> golden_in_OD2 = {
        TEST_FILE("pamir/ObjectDetect/input_QVGA.bin"),
    };
    const std::vector<std::string> golden_outs_OD2 = {
        TEST_FILE("pamir/ObjectDetect/out_QVGA_11.bin"),
        TEST_FILE("pamir/ObjectDetect/out_QVGA_12.bin"),
        TEST_FILE("pamir/ObjectDetect/out_QVGA_13.bin"),
        TEST_FILE("pamir/ObjectDetect/out_QVGA_14.bin"),
        TEST_FILE("pamir/ObjectDetect/out_QVGA_15.bin"),
        TEST_FILE("pamir/ObjectDetect/out_QVGA_16.bin"),
        TEST_FILE("pamir/ObjectDetect/out_QVGA_17.bin"),
        TEST_FILE("pamir/ObjectDetect/out_QVGA_18.bin"),
        TEST_FILE("pamir/ObjectDetect/out_QVGA_19.bin"),
        TEST_FILE("pamir/ObjectDetect/out_QVGA_20.bin"),
    };
    do_test(file_OD2, golden_in_OD2, golden_outs_OD2);

    return 0;
}
