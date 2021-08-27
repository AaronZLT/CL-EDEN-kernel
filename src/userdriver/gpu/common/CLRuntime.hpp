/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

#ifndef USERDRIVER_GPU_CL_OPERATORS_CL_RUNTIME_HPP_
#define USERDRIVER_GPU_CL_OPERATORS_CL_RUNTIME_HPP_

#include <queue>
#include "userdriver/gpu/common/CLBuffer.hpp"
#include "userdriver/gpu/common/CLIncludes.hpp"
#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/common/CLPlatform.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#define MAXLEN_DEVICE_NAME 1024

namespace enn {
namespace ud {
namespace gpu {

// ToDo(all): reduce LOC score
class CLRuntime {
public:
    friend class CLTensor;

    CLRuntime() = default;
    Status initialize(const uint32_t &device_id);
    Status release();

    std::shared_ptr<CLBuffer> getBuffer(const PrecisionType &precision,
                                        const DataType &data_type,
                                        const Dim4 &dim,
                                        const BufferType &buffer_type,
                                        const StorageType &storage_type = StorageType::BUFFER);
    Status assignBufferPool();
    Status resetIntraBuffer();
    Status resetInterBuffer(const std::shared_ptr<CLBuffer> buffer);

    bool isFP16Support(void);
    bool isBifrost(void);
    bool isMakalu(void);
    bool isArmDotAccSupport(void);
    bool isArmDotSupport(void);
    bool isValhall(void);
    bool isAMD(void);

    cl_device_id getDeviceID(void);
    std::string getDeviceName(void);
    cl_context getContext(void);
    cl_command_queue getQueue(void);
    cl_program getProgram(void);

    Status copyBuffer(cl_mem dst, cl_mem src, size_t dst_offset_bytes, size_t src_offset_bytes, size_t size_bytes);
    Status writeBuffer(cl_mem dst, void *src, size_t type_bytes, uint32_t num, cl_bool blocking = CL_TRUE);
    Status readBuffer(void *dst,
                      cl_mem src,
                      size_t type_bytes,
                      uint32_t num,
                      cl_bool blocking = CL_TRUE,
                      void *event = nullptr);
    Status writeBufferTexture2D(cl_mem dst, void *src, Dim4 &dim, cl_bool blocking = CL_TRUE);
    Status readBufferTexture2D(void *dst, cl_mem src, Dim4 &dim, cl_bool blocking = CL_TRUE);

    Status copyFloat2Half(cl_mem dst, cl_mem src, const uint32_t &num);
    Status copyHalf2Float(cl_mem dst, cl_mem src, const uint32_t &num);
    Status copyInt2Float(cl_mem dst, cl_mem src, const uint32_t &num);
    Status copyFloat2Int(cl_mem dst, cl_mem src, const uint32_t &num);
    Status copyFloat2HalfTexture2D(cl_mem dst, cl_mem src, Dim4 &dim);
    Status copyHalf2FloatTexture2D(cl_mem dst, cl_mem src, Dim4 &dim);

    Status NCHW2NHWC(cl_mem src, cl_mem dst, const Dim4 &dim, DataType type, PrecisionChangeMode mode);
    Status NHWC2NCHW(cl_mem src, cl_mem dst, const Dim4 &dim, DataType type, PrecisionChangeMode mode);
    Status NCHW2DHWC4(cl_mem src, cl_mem dst, const Dim4 &dim, DataType type, PrecisionChangeMode mode);
    Status NHWC2DHWC4(cl_mem src, cl_mem dst, const Dim4 &dim, DataType type, PrecisionChangeMode mode);
    Status DHWC42NCHW(cl_mem src, cl_mem dst, const Dim4 &dim, DataType type, PrecisionChangeMode mode);
    Status DHWC42NHWC(cl_mem src, cl_mem dst, const Dim4 &dim, DataType type, PrecisionChangeMode mode);

    // a template for set ocl kernel arguments its a recurrent call
    template <typename T, typename... Args>
    Status setKernelArgId(const cl_kernel &kernel, int id, const T &t, Args... args) {
        int err = clSetKernelArg(kernel, id, sizeof(T), &(t));
        CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "setKernelArgId() fail: %d (id: %d)\n", err, id);
        id++;
        return setKernelArgId(kernel, id, args...);
    }

    Status setKernelByName(std::shared_ptr<struct _cl_kernel> *kernel, const std::string &kernel_name);

    Status setKernel(std::shared_ptr<struct _cl_kernel> *kernel, const std::string &name, const PrecisionType &type);

    Status setCommonKernels();
    Status setPublicBufferKernels();
    Status setPublicTexture2dKernels();

    Status setKernelArgId(const cl_kernel & /*kernel*/, int /*id*/) { return Status::SUCCESS; }

    template <typename... Args> Status setKernelArg(const cl_kernel &kernel, Args... args) {
        return setKernelArgId(kernel, 0, args...);
    }

    Status enqueueKernel(const cl_kernel &kernel,
                         const cl_uint &work_dim,
                         const size_t *const &global_work_size,
                         const size_t *const &local_work_size);

    size_t getRuntimeTypeBytes(const PrecisionType &precision) {
        switch (precision) {
        case PrecisionType::FP32: return sizeof(float);
        case PrecisionType::FP16: return sizeof(float) / 2;
        case PrecisionType::INT8:
        case PrecisionType::UINT8: return sizeof(uint8_t);
        default: DEBUG_PRINT("illeage runtime precision. only FLOAT32 FLOAT16 INT8 are supported.\n");
        }
        return 0;
    }

    uint32_t getComputeUnitsCount() const { return compute_units_count_; }

    size_t *getMaxWorkGroupSize() { return max_work_group_size_.data(); }

    Status GetKernelMaxWorkGroupSize(cl_kernel kernel, cl_device_id device_id, int *result);

    ~CLRuntime();
    // TODO(zhe123.wang): int size -> size_t size
    Status zeroBuf(const int &size, cl_mem buf);
    Status zeroTexture2D(const TextureDescriptor &texture_descriptor, cl_mem buf);

    Status initializeQueue();

private:
    std::deque<std::tuple<uint32_t, bool, std::shared_ptr<CLBuffer>>> intra_buffers;  //  (bytes, used, buffer)
    std::vector<std::pair<std::shared_ptr<CLBuffer>, bool>> inter_buffers;            //  (buffer, used)

    std::mutex mutex_;
    Status initializeDevice(const uint32_t &target_device_id);
    Status initializeContext();
    // Status initializeQueue();
    Status initializeProgram();
    Status initializeProgramFromSource();
    Status initializeProgramKernelSources();
    void printProgramBuildInfo(cl_program program);
    Status preCompileKernels();
    Status genOriginalKernelStringCRC();
    Status genFinalKernelString();
    Status tryInitializeProgramFromBinary(std::string path);
    Status generateBinaryKernel(std::string path, std::string name);
    Status copyBinaryKernel(std::string src, std::string dest);
    Status checkAccessiblePath(std::string path);

    std::string getKernelBinaryDir();
    std::string getKernelBinaryName();
    const std::string &getKernelSourceByName(const std::string &kernel_name);

    cl_channel_type ToImageChannelType(PrecisionType precision);
    cl_mem CreateImage2DLegacy(cl_context context,
                               cl_mem_flags flags,
                               const cl_image_format *image_format,
                               const cl_image_desc *image_desc,
                               void *host_ptr,
                               cl_int *errcode_ret);
    cl_mem allocBuffer(const uint32_t &bytes, const bool &zero_init);
    cl_mem allocBuffer(const uint32_t &bytes, const bool &zero_init, DataPtr data);
    cl_mem allocTexture2D(const TextureDescriptor &texture_descriptor, const bool &zero_init = false);
    Status releaseBuffer(std::shared_ptr<CLBuffer> buffer);

    bool is_bifrost_support_ = false;
    bool is_makalu_support_ = false;
    bool is_fp16_support_ = false;
    bool is_arm_dot_acc_support_ = false;
    bool is_arm_dot_support_ = false;
    bool is_valhall_support_ = false;
    bool is_amd_ = false;

    std::shared_ptr<CLPlatform> platform_;
    cl_device_id *selected_device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    std::vector<std::string> final_kernel_strings_;
    std::string final_all_kernel_;
    std::string kernel_header_;
    std::vector<cl_program> programs_;
    bool is_online_compile_ = false;

    int str_len_cur_ = 0;
    int kernel_bin_version_ = 0;
    std::map<std::string, std::shared_ptr<_cl_kernel>> kernels_;
    std::shared_ptr<struct _cl_kernel> kernel_half2float_;
    std::shared_ptr<struct _cl_kernel> kernel_float2half_;
    std::shared_ptr<struct _cl_kernel> kernel_int2float_;
    std::shared_ptr<struct _cl_kernel> kernel_float2int_;
    std::shared_ptr<struct _cl_kernel> broadcast_kernel_;
    std::shared_ptr<struct _cl_kernel> kernel_NCHW2NHWC_fp16_;
    std::shared_ptr<struct _cl_kernel> kernel_NCHW2NHWC_fp32_;
    std::shared_ptr<struct _cl_kernel> kernel_NCHW2NHWC_int8_;
    std::shared_ptr<struct _cl_kernel> kernel_NCHW2NHWC_uint8_;
    std::shared_ptr<struct _cl_kernel> kernel_NCHW2NHWC_fp322fp16_;
    std::shared_ptr<struct _cl_kernel> kernel_NCHW2NHWC_fp162fp32_;
    std::shared_ptr<struct _cl_kernel> kernel_NCHW2NHWC_int16_;
    std::shared_ptr<struct _cl_kernel> kernel_NCHW2NHWC_uint16_;
    std::shared_ptr<struct _cl_kernel> kernel_NHWC2NCHW_fp16_;
    std::shared_ptr<struct _cl_kernel> kernel_NHWC2NCHW_fp32_;
    std::shared_ptr<struct _cl_kernel> kernel_NHWC2NCHW_int8_;
    std::shared_ptr<struct _cl_kernel> kernel_NHWC2NCHW_uint8_;
    std::shared_ptr<struct _cl_kernel> kernel_NHWC2NCHW_fp322fp16_;
    std::shared_ptr<struct _cl_kernel> kernel_NHWC2NCHW_fp162fp32_;
    std::shared_ptr<struct _cl_kernel> kernel_NHWC2NCHW_int16_;
    std::shared_ptr<struct _cl_kernel> kernel_NHWC2NCHW_uint16_;

    std::shared_ptr<struct _cl_kernel> kernel_NHWC2DHWC4_fp16_;
    std::shared_ptr<struct _cl_kernel> kernel_NHWC2DHWC4_fp32_;
    std::shared_ptr<struct _cl_kernel> kernel_NHWC2DHWC4_fp162fp32_;
    std::shared_ptr<struct _cl_kernel> kernel_NHWC2DHWC4_fp322fp16_;

    std::shared_ptr<struct _cl_kernel> kernel_NCHW2DHWC4_fp16_;
    std::shared_ptr<struct _cl_kernel> kernel_NCHW2DHWC4_fp32_;
    std::shared_ptr<struct _cl_kernel> kernel_NCHW2DHWC4_fp162fp32_;
    std::shared_ptr<struct _cl_kernel> kernel_NCHW2DHWC4_fp322fp16_;

    std::shared_ptr<struct _cl_kernel> kernel_DHWC42NHWC_fp16_;
    std::shared_ptr<struct _cl_kernel> kernel_DHWC42NHWC_fp32_;
    std::shared_ptr<struct _cl_kernel> kernel_DHWC42NHWC_fp162fp32_;
    std::shared_ptr<struct _cl_kernel> kernel_DHWC42NHWC_fp322fp16_;

    std::shared_ptr<struct _cl_kernel> kernel_DHWC42NCHW_fp16_;
    std::shared_ptr<struct _cl_kernel> kernel_DHWC42NCHW_fp32_;
    std::shared_ptr<struct _cl_kernel> kernel_DHWC42NCHW_fp162fp32_;
    std::shared_ptr<struct _cl_kernel> kernel_DHWC42NCHW_fp322fp16_;
    std::shared_ptr<struct _cl_kernel> kernel_half2float_texture2d_;
    std::shared_ptr<struct _cl_kernel> kernel_float2half_texture2d_;

    std::vector<std::string> removeSpecificKernel(const std::vector<std::string> kernel_string,
                                                  const std::string &kernel_name);
    Status preCreateOpenCLKernel(const std::vector<std::string> &opencl_kernels);

    uint32_t compute_units_count_ = 0;
    std::vector<size_t> max_work_group_size_;
};  // class CLRuntime

}  // namespace gpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_GPU_CL_OPERATORS_CL_RUNTIME_HPP_