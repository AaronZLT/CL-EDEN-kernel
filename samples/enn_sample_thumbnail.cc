#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include "enn_api-type.h"
#include "enn_api-public.hpp"
#include "enn_sample_utils.hpp"
#define CMP_OUT

#pragma pack(push, 1)
typedef struct {
    uint32_t frame_width;
    uint32_t frame_height;
    uint32_t isp_gain;
    uint32_t minPatchDifference;
    uint32_t minPatchValueForDiff;
    uint32_t maxPatchValueForDiff;
    uint32_t minPatchDifferenceForLocal;
    uint32_t minPatchValueForDiffForLocal;
    uint32_t maxPatchValueForDiffForLocal;
} thumbnail_param_t;
#pragma pack(pop)

const std::string model_file("/data/raw/thumbnail/thumbnail_64x64.cgo");

const std::vector<std::string> input_files = {
    std::string("/data/raw/thumbnail/thumbnail_input_AE.raw"),
    std::string("/data/raw/thumbnail/thumbnail_input_AWB.raw"),
    std::string("/data/raw/thumbnail/thumbnail_input_PrevFrameArray.raw"),
};

const std::vector<std::string> output_golden = {
    std::string("/data/raw/thumbnail/thumbnail_ref_Y14.raw"),
    std::string("/data/raw/thumbnail/thumbnail_ref_Y12.raw"),
    std::string("/data/raw/thumbnail/thumbnail_ref_Br12.raw"),
    std::string("/data/raw/thumbnail/thumbnail_ref_Br14.raw"),
    std::string("/data/raw/thumbnail/thumbnail_ref_PatchCovariance.raw"),
    std::string("/data/raw/thumbnail/thumbnail_ref_PatchColor.raw"),
    std::string("/data/raw/thumbnail/thumbnail_ref_PatchRGB.raw"),
    std::string("/data/raw/thumbnail/thumbnail_ref_values.raw"),
    std::string("/data/raw/thumbnail/thumbnail_ref_CurrFrameArray.raw"),
};

enum {
    IN_AE = 0,    // input
    IN_AWB,
    IN_PrevFrameArray,
    IN_MAX,
};

enum {
    OUT_Br12 = 0,  // output
    OUT_Br14,
    OUT_CurrFrameArray,
    OUT_PatchColor,
    OUT_PatchCovariance,
    OUT_PatchRGB,
    OUT_Y12,
    OUT_Y14,
    OUT_MAX,
};

int main() try {
    EnnModelId model_id;
    EnnBufferPtr *buffer_set = {};
    NumberOfBuffersInfo buffer_info;
    thumbnail_param_t parameter = { 4032, 3024, 150, 12, 0, 0xFFF, 12, 0, 0xFFF };  // default

    enn::api::EnnInitialize();

    if (enn::api::EnnOpenModel(model_file.c_str(), &model_id))
        throw std::invalid_argument("Open Error");
    if (enn::api::EnnAllocateAllBuffers(model_id, &buffer_set, &buffer_info, 0, false))
        throw std::invalid_argument("Allocate buffer error");

    int n_buffers = buffer_info.n_in_buf + buffer_info.n_out_buf;

    std::cout << " # of buffer: In(" << buffer_info.n_in_buf << "): Out(" << buffer_info.n_out_buf << ")" << std::endl;

    for (int i = 0; i < n_buffers; i++) {
        printf("[%d] va: %p, size: %d, offset: %d\n", i, buffer_set[i]->va, buffer_set[i]->size, buffer_set[i]->offset);
    }

    /* Set parameter buffer */
    EnnBufferPtr param_buf;
    if (enn::api::EnnCreateBuffer(&param_buf, sizeof(parameter)))
        throw std::invalid_argument("Param allocate failed");

    memcpy(param_buf->va, reinterpret_cast<void *>(&parameter), sizeof(parameter));
    enn::api::EnnSetBufferByLabel(model_id, "thumbnail_0", param_buf);


    EnnBufferPtr shapeInfosBuf;
    if (enn::api::EnnCreateBuffer(&shapeInfosBuf, 492))
        throw std::invalid_argument("Param allocate failed");

    memset(shapeInfosBuf->va, 0, 492);
    enn::api::EnnSetBufferByLabel(model_id, "Shape Infos Buffer", shapeInfosBuf);


    /* Copy sample files */
    if (enn::sample_utils::import_file_to_mem(input_files[IN_AE].c_str(), reinterpret_cast<char *>(buffer_set[0]->va)))
        throw std::invalid_argument("Param allocate failed: IN_AE");
    if (enn::sample_utils::import_file_to_mem(input_files[IN_AWB].c_str(), reinterpret_cast<char *>(buffer_set[1]->va)))
        throw std::invalid_argument("Param allocate failed: IN_AWB");
    if (enn::sample_utils::import_file_to_mem(input_files[IN_PrevFrameArray].c_str(), reinterpret_cast<char *>(buffer_set[2]->va)))
        throw std::invalid_argument("Param allocate failed: IN_PrevFrameArray");

    enn::api::EnnBufferCommit(model_id);
    enn::api::EnnExecuteModel(model_id);

  // commented: Check output
    for (int out_idx = 0; out_idx < output_golden.size(); ++out_idx) {
#ifdef CMP_OUT
        EnnBufferPtr out_buf;
        enn::api::EnnCreateBuffer(&out_buf, buffer_set[3 + out_idx]->size);
        enn::sample_utils::import_file_to_mem(output_golden[out_idx].c_str(), reinterpret_cast<char *>(out_buf->va));
        enn::sample_utils::CompareBuffersWithThreshold<uint8_t>(out_buf->va, buffer_set[3 + out_idx]->va, out_buf->size);
        enn::api::EnnReleaseBuffer(out_buf);
#else
        // commented:: file out
        if (enn::sample_utils::export_mem_to_file((output_golden[out_idx] + std::string("_out")).c_str(), buffer_set[3 + out_idx]->va,
                                                 buffer_set[3 + out_idx]->size))
            throw std::invalid_argument("Output export occurs an error");
#endif
    }

    enn::api::EnnReleaseBuffer(param_buf);
    enn::api::EnnReleaseBuffers(buffer_set, n_buffers);

    enn::api::EnnCloseModel(model_id);

    enn::api::EnnDeinitialize();
    // EnnSetBufferByLabel

    return 0;
} catch (std::invalid_argument &e) {
    std::cout << "Error occured: " << e.what() << std::endl;
} catch (std::exception &e) {
    std::cout << "Exception occured: " << e.what() << std::endl;
}
