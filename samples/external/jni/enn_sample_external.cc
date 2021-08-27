#include <iostream>
#include "enn_api-public.hpp"
#include "enn_sample_utils.hpp"

void setPreperence() {
    // Set PerfMode (Default: ENN_PREF_MODE_BOOST_ON_EXE)
    uint32_t perf_mode = ENN_PREF_MODE_NORMAL;
    enn::api::EnnSetPreferencePerfMode(perf_mode);
}

/**
 * Using case1) Allocate all buffers
 */
int enn_sample_1(const std::string filename, const std::vector<std::string> &golden_inputs,
                 const std::vector<std::string> &golden_outputs) {
    PRINT(YELLOW "[enn_sample_1]" RESET);

    if (enn::api::EnnInitialize()) {
        PRINT_AND_RETURN("Initialize Failed");
    }

    // NOTE) If need to set preperence, call preperence APIs before EnnOpenModel()
    setPreperence();

    EnnModelId model_id;

    // Open model and get model_id
    if (enn::api::EnnOpenModel(filename.c_str(), &model_id)) {
        PRINT_AND_RETURN("Open Model Failed: %s", filename.c_str());
    } else {
        PRINT("# Filename: %s", filename.c_str());
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
    if (enn::api::EnnAllocateAllBuffers(model_id, &buffer_set, &buffers_info)) {
        PRINT_AND_RETURN("Allocate Buffer Error");
    }

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

    PRINT("");

    return 0;
}

/**
 * Using case2) Allocate each memory buffers
 */
int enn_sample_2(const std::string filename, const std::vector<std::string> &golden_inputs,
                 const std::vector<std::string> &golden_outputs) {
    PRINT(YELLOW "[enn_sample_2]" RESET);

    if (enn::api::EnnInitialize()) {
        PRINT_AND_RETURN("Initialize Failed");
    }

    setPreperence();

    EnnModelId model_id;

    // Open model and get model_id
    if (enn::api::EnnOpenModel(filename.c_str(), &model_id)) {
        PRINT_AND_RETURN("Open Model Failed: %s", filename.c_str());
    } else {
        PRINT("# Filename: %s", filename.c_str());
    }

    NumberOfBuffersInfo buffers_info;

    if (enn::api::EnnGetBuffersInfo(&buffers_info, model_id)) {
        PRINT_AND_RETURN("Get Buffer information Error");
    }

    if (enn::sample_utils::check_valid_buffers_info(buffers_info, golden_inputs.size(), golden_outputs.size())) {
        PRINT_AND_RETURN("Invalid Buffer information");
    }

    uint32_t n_in_buf = buffers_info.n_in_buf;
    uint32_t n_out_buf = buffers_info.n_out_buf;
    EnnBufferPtr *in_buffers = new EnnBufferPtr[n_in_buf];
    EnnBufferPtr *out_buffers = new EnnBufferPtr[n_out_buf];
    EnnBufferInfo tmp_buf_info;

    for (int i = 0; i < n_in_buf; ++i) {
        if (enn::api::EnnGetBufferInfoByIndex(&tmp_buf_info, model_id, ENN_DIR_IN, i)) {
            PRINT_AND_RETURN("Error: Get Input Buffer information");
        }
        if (enn::api::EnnCreateBuffer(&(in_buffers[i]), tmp_buf_info.size)) {
            PRINT_AND_RETURN("Error: Create Input Buffer");
        }
        if (enn::api::EnnSetBufferByIndex(model_id, ENN_DIR_IN, i, in_buffers[i])) {
            PRINT_AND_RETURN("Error: Set Input Buffer");
        }
    }
    for (int i = 0; i < n_out_buf; ++i) {
        if (enn::api::EnnGetBufferInfoByIndex(&tmp_buf_info, model_id, ENN_DIR_OUT, i)) {
            PRINT_AND_RETURN("Error: Get Output Buffer information");
        }
        if (enn::api::EnnCreateBuffer(&(out_buffers[i]), tmp_buf_info.size)) {
            PRINT_AND_RETURN("Error: Create Output Buffer");
        }
        if (enn::api::EnnSetBufferByIndex(model_id, ENN_DIR_OUT, i, out_buffers[i])) {
            PRINT_AND_RETURN("Error: Set Output Buffer");
        }
    }

    // Load input files to input buffers
    if (enn::sample_utils::load_input_files(0, n_in_buf, golden_inputs, in_buffers)) {
        PRINT_AND_RETURN("Input file load error");
    }

    // Send allocated buffers to service core
    if (enn::api::EnnBufferCommit(model_id)) {
        PRINT_AND_RETURN("Failed to commit buffers");
    }

    // Execute model with commeted buffers
    if (enn::api::EnnExecuteModel(model_id)) {
        PRINT_AND_RETURN("Failed to execute model");
    }

    // Load golden output files to reference buffer and compare with output bufferss
    if (enn::sample_utils::load_golden_files_and_compare(0, n_out_buf, golden_outputs, out_buffers)) {
        PRINT_AND_RETURN("Golden output file load error");
    }

    // Release buffers
    for (int i = 0; i < n_in_buf; ++i) {
        if (enn::api::EnnReleaseBuffer(in_buffers[i])) {
            PRINT_AND_RETURN("Failed to Release Input buffers");
        }
    }
    for (int i = 0; i < n_out_buf; ++i) {
        if (enn::api::EnnReleaseBuffer(out_buffers[i])) {
            PRINT_AND_RETURN("Failed to Release Output buffers");
        }
    }

    delete[] in_buffers;
    delete[] out_buffers;

    // Close model
    if (enn::api::EnnCloseModel(model_id)) {
        PRINT_AND_RETURN("Failed to close model");
    }

    if (enn::api::EnnDeinitialize()) {
        PRINT_AND_RETURN("Deinitialize Failed");
    }

    PRINT("");

    return 0;
}

/**
 * Using case3) Allocate all buffers with N sessions
 */
int enn_sample_3(const std::string filename, const std::vector<std::string> &golden_inputs,
                 const std::vector<std::string> &golden_outputs) {
    PRINT(YELLOW "[enn_sample_3]" RESET);

    const int session_N = 2;

    PRINT(CYAN "Session num : %d" RESET, session_N);

    if (enn::api::EnnInitialize()) {
        PRINT_AND_RETURN("Initialize Failed");
    }

    setPreperence();

    EnnModelId model_id;

    // Open model and get model_id
    if (enn::api::EnnOpenModel(filename.c_str(), &model_id)) {
        PRINT_AND_RETURN("Open Model Failed: %s", filename.c_str());
    } else {
        PRINT("# Filename: %s", filename.c_str());
    }

    // generate buffer space before calling EnnAllocateAllBuffers()
    if (enn::api::EnnGenerateBufferSpace(model_id, session_N + 1)) {
        PRINT_AND_RETURN("Failed to generate buffer space");
    }

    EnnBufferPtr *buffer_set[session_N];
    NumberOfBuffersInfo buffers_info[session_N];

    for (int si = 0; si < session_N; ++si) {
        PRINT(CYAN "Session[%d] allocate & load" RESET, si);

        if (enn::api::EnnAllocateAllBuffers(model_id, &buffer_set[si], &buffers_info[si], si)) {
            PRINT_AND_RETURN("Allocate Buffer Error");
        }

        if (enn::sample_utils::check_valid_buffers_info(buffers_info[si], golden_inputs.size(), golden_outputs.size())) {
            PRINT_AND_RETURN("Invalid Buffer information");
        }

        uint32_t n_in_buf = buffers_info[si].n_in_buf;

        // Load input files to input buffers
        int in_idx_start = 0;
        int in_idx_end = n_in_buf;
        if (enn::sample_utils::load_input_files(in_idx_start, in_idx_end, golden_inputs, buffer_set[si])) {
            PRINT_AND_RETURN("Input file load error");
        }
    }

    for (int si = 0; si < session_N; ++si) {
        // Execute model with commeted buffers
        if (enn::api::EnnExecuteModel(model_id, si)) {
            PRINT_AND_RETURN("Failed to execute model");
        }
    }

    for (int si = 0; si < session_N; ++si) {
        PRINT(CYAN "Session[%d] golden compare & release" RESET, si);

        uint32_t n_in_buf = buffers_info[si].n_in_buf;
        uint32_t n_out_buf = buffers_info[si].n_out_buf;

        // Load golden output files to reference buffer and compare with output bufferss
        int out_idx_start = n_in_buf;
        int out_idx_end = out_idx_start + static_cast<uint32_t>(golden_outputs.size());
        if (enn::sample_utils::load_golden_files_and_compare(out_idx_start, out_idx_end, golden_outputs, buffer_set[si])) {
            PRINT_AND_RETURN("Golden output file load error");
        }

        // Release buffers
        if (enn::api::EnnReleaseBuffers(buffer_set[si], n_in_buf + n_out_buf)) {
            PRINT_AND_RETURN("Failed to Release buffers");
        }
    }

    // Close model
    if (enn::api::EnnCloseModel(model_id)) {
        PRINT_AND_RETURN("Failed to close model");
    }

    if (enn::api::EnnDeinitialize()) {
        PRINT_AND_RETURN("Deinitialize Failed");
    }

    PRINT("");

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

    enn_sample_1(filename, golden_in, golden_out);

    enn_sample_2(filename, golden_in, golden_out);

    enn_sample_3(filename, golden_in, golden_out);

    return 0;
}
