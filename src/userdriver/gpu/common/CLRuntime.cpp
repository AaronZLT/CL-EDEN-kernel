#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iostream>

#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/common/eden_osal/log.h"
//TODO(xin.lu): remove useless code and EDEN log
namespace enn {
namespace ud {
namespace gpu {
#define POLY 0x1021
#define SYS32 4  // sizeof(long) equal 4 in 32bit system
#define SYS64 8  // sizeof(long) equal 8 in 64bit system
int crc16(const char *addr, int num, int crc) {
    int i;
    for (; num > 0; num--) /* Step through bytes in memory */
    {
        crc = crc ^ (*addr++ << 8); /* Fetch byte from memory, XOR into CRC top byte*/
        for (i = 0; i < 8; i++)     /* Prepare to rotate 8 bits */
        {
            if (crc & 0x8000)            /* b15 is set... */
                crc = (crc << 1) ^ POLY; /* rotate and XOR with polynomic */
            else                         /* b15 is clear... */
                crc <<= 1;               /* just rotate */
        }                                /* Loop for 8 bits */
        crc &= 0xFFFF;                   /* Ensure CRC remains 16-bit value */
    }                                    /* Loop until num=0 */
    return (crc);                        /* Return updated CRC */
}

std::shared_ptr<CLBuffer> CLRuntime::getBuffer(const PrecisionType &precision,
                                               const DataType &data_type,
                                               const Dim4 &dims,
                                               const BufferType &buffer_type,
                                               const StorageType &storage_type) {
    if (storage_type == StorageType::BUFFER) {
        size_t bytes =
            getTypeBytes(data_type, precision) * (size_t)dims.n * (size_t)dims.c * (size_t)dims.h * (size_t)dims.w;
        switch (buffer_type) {
            case BufferType::DEDICATED: {
                std::shared_ptr<CLBuffer> buffer = std::make_shared<CLBuffer>(bytes);
                buffer->assignBuffer(allocBuffer(bytes, true));
                return buffer;
            }
            case BufferType::INTER_SHARED_NEW: {  // assign new buffer for input of operator
                std::shared_ptr<CLBuffer> buffer = std::make_shared<CLBuffer>(bytes);
                buffer->assignBuffer(nullptr);
                inter_buffers.push_back(std::make_pair(buffer, true));
                return buffer;
            }
            case BufferType::INTER_SHARED_REUSE: {  // reuse buffer for output of operator
                // reuse buffer
                for (auto &buffer : inter_buffers) {
                    if (!buffer.second) {
                        if (buffer.first->getBytes() < bytes) {
                            buffer.first->setBytes(bytes);
                        }
                        buffer.second = true;
                        return buffer.first;
                    }
                }
                // assign new buffer
                std::shared_ptr<CLBuffer> buffer = std::make_shared<CLBuffer>(bytes);
                buffer->assignBuffer(nullptr);
                inter_buffers.push_back(std::make_pair(buffer, true));
                return buffer;
            }
            case BufferType::INTRA_SHARED: {
                DEBUG_PRINT("Current Intra Buffer Pool size is %zu", intra_buffers.size());
                // check if there is free buffer that is greater than required size
                for (size_t i = 0; i < intra_buffers.size(); i++) {
                    if (!std::get<1>(intra_buffers[i]) && std::get<0>(intra_buffers[i]) >= bytes) {
                        DEBUG_PRINT("Reusing buffer");
                        std::get<1>(intra_buffers[i]) = true;
                        return std::get<2>(intra_buffers[i]);
                    }
                }

                // check if there is free buffer that is smaller than required size
                for (int32_t i = static_cast<int32_t>(intra_buffers.size()) - 1; i >= 0; i--) {
                    if (!std::get<1>(intra_buffers[i])) {
                        DEBUG_PRINT("Reusing buffer with larger size");
                        std::get<0>(intra_buffers[i]) = bytes;
                        std::get<1>(intra_buffers[i]) = true;
                        const auto mem = std::get<2>(intra_buffers[i]);
                        sort(intra_buffers.begin(), intra_buffers.end());
                        return mem;
                    }
                }

                DEBUG_PRINT("Assigning new buffer");
                std::shared_ptr<CLBuffer> buffer = std::make_shared<CLBuffer>(0);
                buffer->assignBuffer(nullptr);
                intra_buffers.push_back(std::make_tuple(bytes, true, buffer));
                sort(intra_buffers.begin(), intra_buffers.end());
                return buffer;
            }
            default:
                ERROR_PRINT("Error in assigning buffer");
        }
    } else {
        TextureDescriptor texture_descriptor;
        const int slices = IntegralDivideRoundUp(dims.c, 4);

        texture_descriptor.image_height = dims.h * slices;
        texture_descriptor.image_width = dims.w * dims.n;
        texture_descriptor.bytes = getTypeBytes(data_type, precision) * (size_t)(texture_descriptor.image_height) * (size_t)(texture_descriptor.image_width);
        texture_descriptor.precision = precision;

        std::shared_ptr<CLBuffer> buffer = std::make_shared<CLBuffer>(texture_descriptor.bytes);
        buffer->assignBuffer(allocTexture2D(texture_descriptor));
        return buffer;
    }
    return NULL;
}

Status CLRuntime::assignBufferPool() {
    DEBUG_PRINT("Intra Buffer Pool size is %zd", intra_buffers.size());
    for (auto buffer = intra_buffers.begin(); buffer != intra_buffers.end();) {
        DEBUG_PRINT("Intra Buffer size is %fMB", std::get<0>(*buffer) / 1024.0 / 1024.0);
        std::get<2>(*buffer)->setBytes(std::get<0>(*buffer));
        std::get<2>(*buffer)->assignBuffer(allocBuffer(std::get<0>(*buffer), true));
        buffer = intra_buffers.erase(buffer);
    }

    DEBUG_PRINT("Inter Buffer Pool size is %zd", inter_buffers.size());
    for (auto buffer_pair = inter_buffers.begin(); buffer_pair != inter_buffers.end();) {
        DEBUG_PRINT("Inter Buffer size is %fMB",
                    (*buffer_pair).first->getBytes() / 1024.0 / 1024.0);
        (*buffer_pair).first->assignBuffer(allocBuffer((*buffer_pair).first->getBytes(), false));
        buffer_pair = inter_buffers.erase(buffer_pair);
    }
    return Status::SUCCESS;
}

Status CLRuntime::resetIntraBuffer() {
    DEBUG_PRINT("Intra Buffer Pool reset");
    for (auto &buffer : intra_buffers) {
        std::get<1>(buffer) = false;
    }
    return Status::SUCCESS;
}

Status CLRuntime::resetInterBuffer(const std::shared_ptr<CLBuffer> buffer) {
    for (auto &buffer_pair : inter_buffers) {
        if (buffer_pair.first == buffer) {
            buffer_pair.second = false;
            DEBUG_PRINT("Reusing buffer");
            return Status::SUCCESS;
        }
    }
    ERROR_PRINT("Buffer not found");
    return Status::FAILURE;
}

Status CLRuntime::initializeDevice(const uint32_t &target_device_id) {
    cl_uint err = 0;

    cl_device_id *devices = platform_->getDevices();

    char device_name[MAXLEN_DEVICE_NAME];
    device_name[MAXLEN_DEVICE_NAME - 1] = '\0';

    err = clGetDeviceInfo(
        devices[target_device_id], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clGetDeviceInfo() error (1), err: %d", err);

    std::string device_name_str(device_name);
    DEBUG_PRINT("initializeDevices(): [%d: %s]", target_device_id, device_name_str.c_str());

    is_bifrost_support_ = (std::string::npos != device_name_str.find("G52") ||
                           std::string::npos != device_name_str.find("G71") ||
                           std::string::npos != device_name_str.find("G72") ||
                           std::string::npos != device_name_str.find("G76") ||
                           std::string::npos != device_name_str.find("TTRX") ||
                           std::string::npos != device_name_str.find("G77") ||
                           std::string::npos != device_name_str.find("TBEX") ||  // alias of G78
                           std::string::npos != device_name_str.find("G78"));
#if defined(__ANDROID__)

    is_makalu_support_ = (std::string::npos != device_name_str.find("G52") ||
                          std::string::npos != device_name_str.find("G76") ||
                          std::string::npos != device_name_str.find("TTRX") ||
                          std::string::npos != device_name_str.find("G77") ||
                          std::string::npos != device_name_str.find("TBEX") ||
                          std::string::npos != device_name_str.find("G78"));

#elif (__linux__)
    is_makalu_support_ = false;
#endif  // !__linux__

    is_arm_dot_acc_support_ = (std::string::npos != device_name_str.find("TTRX") ||
                               std::string::npos != device_name_str.find("G77") ||
                               std::string::npos != device_name_str.find("TBEX") ||
                               std::string::npos != device_name_str.find("G78"));
    is_arm_dot_support_ =  (std::string::npos != device_name_str.find("TTRX") ||
                            std::string::npos != device_name_str.find("G77") ||
                            std::string::npos != device_name_str.find("TBEX") ||
                            std::string::npos != device_name_str.find("G78"));

    is_valhall_support_ = (std::string::npos != device_name_str.find("TTRX") ||
                           std::string::npos != device_name_str.find("G77") ||
                           std::string::npos != device_name_str.find("TBEX") ||
                           std::string::npos != device_name_str.find("G78"));
    if (PlatformType::AMD == platform_->getPlatformType()) {
        is_valhall_support_ = true;
        is_makalu_support_ = true;
        is_bifrost_support_ = true;
        is_amd_ = true;
    }

    size_t ext_size = 0;
    char *ext_data = NULL;
    err = clGetDeviceInfo(devices[target_device_id], CL_DEVICE_EXTENSIONS, 0, NULL, &ext_size);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clGetDeviceInfo() error (2), err: %d", err);

    if (0 != ext_size) {
        ext_data = static_cast<char *>(malloc(ext_size));
    } else {
        DEBUG_PRINT("clGetDeviceInfo() error");
        return Status::FAILURE;
    }

    err =
        clGetDeviceInfo(devices[target_device_id], CL_DEVICE_EXTENSIONS, ext_size, ext_data, NULL);
    std::string ext_model(ext_data);
    free(ext_data);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clGetDeviceInfo() error (3), err: %d", err);

    is_fp16_support_ = (std::string::npos != ext_model.find("fp16"));

    selected_device_ = &(devices[target_device_id]);

    if (isBifrost()) {
        DEBUG_PRINT("initializeDevices(): Bifrost architecture");
    } else {
        DEBUG_PRINT("initializeDevices(): Not Bifrost architecture");
    }

    if (isFP16Support()) {
        DEBUG_PRINT("initializeDevices(): FP16 is supported");
    } else {
        DEBUG_PRINT("initializeDevices(): FP16 is not supported");
    }

    err = clGetDeviceInfo(devices[target_device_id],
                          CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(uint32_t),
                          &compute_units_count_,
                          nullptr);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clGetDeviceInfo() error (4), err: %d", err);

    int dims_count;

    err = clGetDeviceInfo(devices[target_device_id],
                          CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                          sizeof(int32_t),
                          &dims_count,
                          nullptr);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clGetDeviceInfo() error (5), err: %d", err);

    max_work_group_size_.resize(dims_count);
    err = clGetDeviceInfo(devices[target_device_id],
                          CL_DEVICE_MAX_WORK_ITEM_SIZES,
                          sizeof(size_t) * dims_count,
                          max_work_group_size_.data(),
                          nullptr);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clGetDeviceInfo() error (6), err: %d", err);

    cl_ulong mem_size = 0;
    err =
        clGetDeviceInfo(devices[target_device_id], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                        sizeof(cl_ulong), &mem_size, nullptr);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err,
                              "clGetDeviceInfo() error (7), err: %d", err);
    DEBUG_PRINT("CL_DEVICE_MAX_MEM_ALLOC_SIZE(): %ju M", mem_size/1024/1024);

    return Status::SUCCESS;
}

Status CLRuntime::GetKernelMaxWorkGroupSize(cl_kernel kernel, cl_device_id device_id, int *result) {
    size_t max_work_group_size;
    cl_int error =
        clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(size_t), &max_work_group_size, nullptr);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == error, "clGetKernelWorkGroupInfo() error (6), err: %d", error);
    *result = static_cast<int>(max_work_group_size);
    return Status::SUCCESS;
}

Status CLRuntime::initializeContext() {
    int err = 0;

    cl_platform_id *platform_id = platform_->getPlatform();
    cl_context_properties context_properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)(*platform_id), 0};

    context_ = clCreateContext(context_properties, 1, selected_device_, NULL, NULL, &err);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clCreateContext() error, err: %d\n", err);
    return Status::SUCCESS;
}

Status CLRuntime::genOriginalKernelStringCRC() {
    DEBUG_PRINT("genOriginalKernelStringCRC() is called!\n");
    kernel_header_ =
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
        "#define dot8(a,b) (dot(a.hi,b.hi)+dot(a.lo,b.lo))\n";

    std::string kernel_header_makalu =
        "#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable\n"
        "#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable\n"
        "#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int16 : enable\n";

    if (isMakalu()) {
        kernel_header_ += kernel_header_makalu;
    }
    if (isArmDotAccSupport()) {
        kernel_header_ += "#define ARM_DOT(x, y, val) val = arm_dot_acc((x), (y), (val))\n";
    } else if (isArmDotSupport()) {
        kernel_header_ += "#define ARM_DOT(x, y, val) val += arm_dot((x), (y))\n";
    } else {
        kernel_header_ += "#define ARM_DOT(x, y, val) val += (((x).s0 * (y).s0) + ((x).s1 * (y).s1) + ((x).s2 * (y).s2) + ((x).s3 * (y).s3))\n";
    }

#if !defined(__ANDROID__)
    kernel_header_ += "#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable\n";
#endif

    if (!is_online_compile_) {
        for (uint32_t i = 0; i < NUM_OF_PRIVATE_KERNEL_HEADER; ++i) {
            kernel_header_ += private_kernel_header[i];
        }
    }

    str_len_cur_ = kernel_header_.length() + Kernels::KernelLengthInMap();
    // when developer change the kernel code, this version number should be changed
    kernel_bin_version_ = 1;

    return Status::SUCCESS;
}

Status CLRuntime::genFinalKernelString() {
    DEBUG_PRINT("genFinalKernelString() is called!\n");

    if (!is_online_compile_) {
        // push kernels in map to vector `final_kernel_strings_`
        final_kernel_strings_.reserve(final_kernel_strings_.size() + Kernels::KernelMap().size());
        for (auto iter = Kernels::KernelMap().begin(); iter != Kernels::KernelMap().end(); ++iter) {
            final_kernel_strings_.push_back(iter->second);
        }
    }

    // The "arm_dot" instruction in gemm1xXQuantized convolution kernel is not supported on
    // non-makalu platforms, so the special treatment is needed for makalu device.
    if (!isMakalu()) {
        auto &kernel_strings = final_kernel_strings_;
        kernel_strings = removeSpecificKernel(kernel_strings, "gemm_block4x4");
#if !defined(__ANDROID__)
        kernel_strings = removeSpecificKernel(kernel_strings, "gemmMakalu_FP16");
        kernel_strings = removeSpecificKernel(kernel_strings, "gemmMakalu_FP32");
        // The following kernels match a long "#if\\s+.+?#endif" which exceeds kitt board limit,
        // so they are removed from kernel_strings.
        kernel_strings = removeSpecificKernel(kernel_strings, "direct1x1_8x4_FP32");
        kernel_strings = removeSpecificKernel(kernel_strings, "direct3x3_8x8_FP16");
        kernel_strings = removeSpecificKernel(kernel_strings, "direct5x5_8x8_FP16");
        kernel_strings = removeSpecificKernel(kernel_strings, "direct7x7_8x8_FP16");
        kernel_strings = removeSpecificKernel(kernel_strings, "direct9x9_8x8_FP16");
        kernel_strings = removeSpecificKernel(kernel_strings, "direct3x3_4x4_FP16");

        kernel_strings = removeSpecificKernel(kernel_strings, "direct3x3_8x8_FP32");
        kernel_strings = removeSpecificKernel(kernel_strings, "direct5x5_8x8_FP32");
        kernel_strings = removeSpecificKernel(kernel_strings, "direct7x7_8x8_FP32");
        kernel_strings = removeSpecificKernel(kernel_strings, "direct9x9_8x8_FP32");
        kernel_strings = removeSpecificKernel(kernel_strings, "direct3x3_4x4_FP32");
#endif
    }
    return Status::SUCCESS;
}

Status CLRuntime::initializeProgramFromSource() {
    DEBUG_PRINT("initializeProgramFromSource() is called!\n");
    LOGI(EDEN_CL, "=== Begin initializeProgramFromSource ===\n");
    int err;
    LOGI(EDEN_CL, "Step 1: begin generate final all kernel string...\n");
    genFinalKernelString();
    LOGI(EDEN_CL, "Finish generate final all kernel string\n");
    final_all_kernel_ = kernel_header_;
    for (auto iter = Kernels::KernelMap().begin(); iter != Kernels::KernelMap().end(); ++iter) {
        final_all_kernel_ += iter->second;
    }
    const char *final_char_source = final_all_kernel_.c_str();
    LOGI(EDEN_CL, "Step 2: begin clCreateProgramWithSource...\n");
    program_ = clCreateProgramWithSource(context_, 1, &final_char_source, NULL, &err);
    LOGI(EDEN_CL, "Finish clCreateProgramWithSource, ret = %d\n", err);
    if (CL_SUCCESS != err) {
        clReleaseContext(context_);
        clReleaseCommandQueue(queue_);
        ERROR_PRINT_RETURN_FAILURE("clCreateProgramWithSource() fail: %d", err);
    }

    const char options[] = "-cl-std=CL1.2 -cl-mad-enable";
    LOGI(EDEN_CL, "Step 3: begin clBuildProgram...\n");
    err = clBuildProgram(program_, 1, selected_device_, options, NULL, NULL);
    LOGI(EDEN_CL, "Finish clBuildProgram, ret = %d\n", err);
    if (err != CL_SUCCESS) {
        printProgramBuildInfo(program_);
        clReleaseContext(context_);
        clReleaseCommandQueue(queue_);
        clReleaseProgram(program_);
        ERROR_PRINT_RETURN_FAILURE("clBuildProgram() fail");
    }
    LOGI(EDEN_CL, "==== End initializeProgramFromSource ===\n");

    return Status::SUCCESS;
}

void CLRuntime::printProgramBuildInfo(cl_program program) {
    size_t logSize = 0;
    clGetProgramBuildInfo(program, selected_device_[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    if (logSize > 1) {
        char *log = new char[logSize];
        clGetProgramBuildInfo(program, selected_device_[0], CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        LOGI(EDEN_CL, "=== Build log: ===\n %s\n", log);
        delete[] log;
    }
}

Status CLRuntime::generateBinaryKernel(std::string filePath, std::string fileName) {
    // binary format is shown as below
    // kernel bin version :                 int(4bytes)
    // kernel string lenth :                int(4bytes)
    // kernel crc :                         int(4bytes)
    // kernel ddk version string size :     int(4bytes)
    // binary kernel string size :          int(4bytes)
    // kernel ddk version :                 string
    // binary kernel :                      string
    DEBUG_PRINT("generateBinaryKernel() is called\n");
    LOGI(EDEN_CL, "=== Begin generateBinaryKernel ===\n");
    cl_int err;
    // create all opencl kernels
    Status state;
    LOGI(EDEN_CL, "Step 1: Begin create OpenCL kernel for all kernel string...\n");
    state = preCreateOpenCLKernel(final_kernel_strings_);
    LOGI(EDEN_CL, "Finish create OpenCL kernel for all kernel string, ret = %d\n", state);
    if (state != Status::SUCCESS) {
        return Status::FAILURE;
    }
    // get binary kernel info
    size_t program_binary_sizes;
    LOGI(EDEN_CL, "Step 2: Begin clGetProgramInfo to get binary size...\n");
    err = clGetProgramInfo(
        program_, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &program_binary_sizes, NULL);
    LOGI(EDEN_CL, "Finish clGetProgramInfo, ret = %d\n", err);
    CHECK_EXPR_RETURN_FAILURE(err == CL_SUCCESS, "clGetProgramInfo() error, err: %d\n", err);
    DEBUG_PRINT("program_binary_sizes :%zu\n", program_binary_sizes);
    unsigned char *program_binaries = new unsigned char[program_binary_sizes];
    LOGI(EDEN_CL, "Step 3: Begin clGetProgramInfo to get binary info...\n");
    err =
        clGetProgramInfo(program_, CL_PROGRAM_BINARIES, sizeof(char *), &(program_binaries), NULL);
    LOGI(EDEN_CL, "Finish clGetProgramInfo, ret = %d\n", err);
    if (err != CL_SUCCESS) {
        delete[] program_binaries;
        ERROR_PRINT_RETURN_FAILURE("clGetProgramInfo() error, err: %d\n", err);
    }
    // get driver info
    size_t driver_str_size;
    LOGI(EDEN_CL, "Step 4: Begin clGetDeviceInfo to get driver string size...\n");
    err = clGetDeviceInfo(*selected_device_, CL_DEVICE_VERSION, 0, NULL, &driver_str_size);
    LOGI(EDEN_CL, "Finish clGetDeviceInfo, ret = %d\n", err);
    if (err != CL_SUCCESS) {
        delete[] program_binaries;
        ERROR_PRINT_RETURN_FAILURE("clGetDeviceInfo() error, err: %d\n", err);
    }
    DEBUG_PRINT("driver_str_size :%zu\n", driver_str_size);
    unsigned char *driver_string = new unsigned char[driver_str_size];
    LOGI(EDEN_CL, "Step 5: Begin clGetDeviceInfo to get driver string...\n");
    err =
        clGetDeviceInfo(*selected_device_, CL_DEVICE_VERSION, driver_str_size, driver_string, NULL);
    LOGI(EDEN_CL, "Finish clGetDeviceInfo, ret = %d\n", err);
    if (err != CL_SUCCESS) {
        delete[] program_binaries;
        delete[] driver_string;
        ERROR_PRINT_RETURN_FAILURE("clGetDeviceInfo() error, err: %d\n", err);
    }
    DEBUG_PRINT("driver_str :%s\n", driver_string);
    // open file
    LOGI(EDEN_CL, "Step 6: Begin make kernel binary file's path -> : %s\n", filePath.c_str());
    if ((access(filePath.c_str(), F_OK) && access(filePath.c_str(), W_OK))) {
        err = mkdir(filePath.c_str(), S_IRWXU | S_IRWXG | S_IROTH);
        LOGI(EDEN_CL, "Finish make kernel binary file's path, ret = %d\n", err);
        if (err == -1) {
            delete[] program_binaries;
            delete[] driver_string;
            ERROR_PRINT_RETURN_FAILURE("create path for binary kernel fail\n");
        }
    }
    LOGI(EDEN_CL, "Step 7: Begin write kernel binary to file -> : %s\n", fileName.c_str());
    std::ofstream kernel_outfile(fileName, std::ofstream::binary);
    if (kernel_outfile.is_open() == false) {
        LOGW(EDEN_CL, "Open binary kernel file failed!\n");
        delete[] program_binaries;
        delete[] driver_string;
        ERROR_PRINT_RETURN_FAILURE("open binary kernel for write fail\n");
    }

    int crc_bin = crc16((const char *)(program_binaries), program_binary_sizes, 0);
    // write file
    kernel_outfile.write((const char *)&kernel_bin_version_, 4);
    LOGW(EDEN_CL, "kernel_outfile.write -> kernel_bin_version_ : %d", kernel_bin_version_);
    kernel_outfile.write((const char *)&str_len_cur_, 4);
    LOGW(EDEN_CL, "kernel_outfile.write -> str_len_cur_ : %d", str_len_cur_);
    kernel_outfile.write((const char *)&driver_str_size, 4);
    LOGW(EDEN_CL, "kernel_outfile.write -> driver_str_size : %zu", driver_str_size);
    kernel_outfile.write((const char *)&program_binary_sizes, 4);
    LOGW(EDEN_CL, "kernel_outfile.write -> program_binary_sizes : %zu", program_binary_sizes);
    kernel_outfile.write((const char *)(driver_string), driver_str_size);
    LOGW(EDEN_CL, "kernel_outfile.write -> driver_string : %s", driver_string);
    kernel_outfile.write((const char *)&crc_bin, 4);
    LOGW(EDEN_CL, "kernel_outfile.write -> crc_bin : %d", crc_bin);
    kernel_outfile.write((const char *)(program_binaries), program_binary_sizes);
    // close file
    kernel_outfile.close();
    chmod(fileName.c_str(), 0664);
    // delete temp memory
    delete[] program_binaries;
    delete[] driver_string;

    LOGI(EDEN_CL, "Finish write kernel binary to file\n");
    LOGI(EDEN_CL, "=== Finish generateBinaryKernel ===\n");
    return Status::SUCCESS;
}

Status CLRuntime::copyBinaryKernel(const std::string src, const std::string dest) {
    DEBUG_PRINT("copyBinaryKernel() is called \n");
    DEBUG_PRINT("src : %s \n", src.c_str());
    DEBUG_PRINT("dest : %s \n", dest.c_str());

    std::ifstream check_dest_binary(dest, std::ifstream::binary);
    if (check_dest_binary.is_open() == true) {
        check_dest_binary.close();
        DEBUG_PRINT("do NOT need to copy gpu kernel\n");
        return Status::SUCCESS;
    } else {
        DEBUG_PRINT("need to copy gpu kernel\n");
        cl_int err = 0;
        std::string dir_name = getKernelBinaryDir();
        if ((access(dir_name.c_str(), F_OK) && access(dir_name.c_str(), W_OK))) {
            err = mkdir(dir_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH);
            if (err == -1) {
                ERROR_PRINT_RETURN_FAILURE("create path for binary kernel fail\n");
            }
        }

        std::ifstream source_file(src, std::ios::binary);
        if (source_file.is_open() == false) {
            ERROR_PRINT_RETURN_FAILURE("open binary kernel for read fail\n");
        }

        std::ofstream dest_file(dest, std::ios::binary);
        if (dest_file.is_open() == false) {
            source_file.close();
            ERROR_PRINT_RETURN_FAILURE("open binary kernel for write fail\n");
        }
        std::istreambuf_iterator<char> begin_source(source_file);
        std::istreambuf_iterator<char> end_source;
        std::ostreambuf_iterator<char> begin_dest(dest_file);
        std::copy(begin_source, end_source, begin_dest);

        source_file.close();
        dest_file.close();
        chmod(dest.c_str(), 0664);
    }

    return Status::SUCCESS;
}

Status CLRuntime::checkAccessiblePath(std::string path) {
    std::ofstream dest_file(path, std::ios::binary | std::ofstream::app);
    if (dest_file.is_open() == false) {
        return Status::FAILURE;
    }
    return Status::SUCCESS;
    // Note that any open file is automatically closed when the ofstream object is destroyed.
}

Status CLRuntime::tryInitializeProgramFromBinary(std::string fileName) {
    DEBUG_PRINT("tryInitializeProgramFromBinary() is called \n");
    DEBUG_PRINT("binary to be used is %s\n", fileName.c_str());
    cl_int err;

    // check binary file
    std::ifstream kernel_binary(fileName, std::ifstream::binary);
    if (kernel_binary.is_open() == false) {
        LOGW(EDEN_CL, "open binary kernel for read fail\n");
        ERROR_PRINT_RETURN_FAILURE("open binary kernel for read fail\n");
    }
    // check binary version
    int bin_ver = 0;
    kernel_binary.read(reinterpret_cast<char *>(&bin_ver), 4);
    if (bin_ver != kernel_bin_version_) {
        LOGW(EDEN_CL,
             "binary kernel version dismatched bin_ver : %d, expected kernel_bin_version_ : %d\n",
             bin_ver,
             kernel_bin_version_);
        ERROR_PRINT_RETURN_FAILURE("binary kernel version dismatched %d \n", bin_ver);
    }
    // check kernel lenth
    int bin_str_len = 0;
    kernel_binary.read(reinterpret_cast<char *>(&bin_str_len), 4);
    if (bin_str_len != str_len_cur_) {
        LOGW(EDEN_CL,
             "binary kernel string lenth dismatched bin_str_len : %d, expected str_len_cur_ : %d\n",
             bin_str_len,
             str_len_cur_);
        ERROR_PRINT_RETURN_FAILURE("binary kernel string lenth dismatched %d \n", bin_str_len);
    }
    // get ddk version
    int bin_ddk_ver_lenth = 0;
    int bin_ker_lenth = 0;
    kernel_binary.read(reinterpret_cast<char *>(&bin_ddk_ver_lenth), 4);
    kernel_binary.read(reinterpret_cast<char *>(&bin_ker_lenth), 4);
    if (bin_ddk_ver_lenth == 0 || bin_ker_lenth == 0) {
        LOGW(EDEN_CL,
             "binary lenth dismatched bin_ddk_ver_lenth : %d, bin_ker_lenth %d\n",
             bin_ddk_ver_lenth,
             bin_ker_lenth);
        ERROR_PRINT_RETURN_FAILURE(
            "binary lenth dismatched %d %d\n", bin_ddk_ver_lenth, bin_ker_lenth);
    }
    char *bin_ddk_string = new char[bin_ddk_ver_lenth];
    kernel_binary.read(bin_ddk_string, bin_ddk_ver_lenth);
    // get driver info
    size_t driver_str_size;
    err = clGetDeviceInfo(*selected_device_, CL_DEVICE_VERSION, 0, NULL, &driver_str_size);
    if (err != CL_SUCCESS) {
        delete[] bin_ddk_string;
        LOGW(EDEN_CL, "clGetDeviceInfo() error, err: %d\n", err);
        ERROR_PRINT_RETURN_CL_FAILURE("clGetDeviceInfo() error, err: %d\n", err);
    }
    DEBUG_PRINT("driver_str_size :%zu\n", driver_str_size);
    unsigned char *driver_string = new unsigned char[driver_str_size];
    err =
        clGetDeviceInfo(*selected_device_, CL_DEVICE_VERSION, driver_str_size, driver_string, NULL);
    if (err != CL_SUCCESS) {
        delete[] driver_string;
        delete[] bin_ddk_string;
        LOGW(EDEN_CL, "clGetDeviceInfo() error, err: %d\n", err);
        ERROR_PRINT_RETURN_CL_FAILURE("clGetDeviceInfo() error, err: %d\n", err);
    }
    LOGI(EDEN_CL, "CL_DEVICE_VERSION : %s\n", driver_string);
    LOGI(EDEN_CL, "CL_DEVICE_BIN_VERSION : %s\n", bin_ddk_string);

    // compare ddk version
    // ddk version str format eg, OpenCL 2.0 v1.r16p0-01rel0.1b894b8f33c9649a67d6a3c1f3f8fea3
    const char *sep = ".";
    char *res_dev;
    char *res_bin;
    char *sep_dev = new char[MAXLEN_DEVICE_NAME];
    char *sep_bin = new char[MAXLEN_DEVICE_NAME];
    sep_dev[0] = '\0';
    sep_bin[0] = '\0';
    int cmp_times = 0;
    res_dev = std::strtok(reinterpret_cast<char *>(driver_string), sep);
    while (res_dev && cmp_times++ < 3) {
        std::strcat(sep_dev, res_dev);
        res_dev = std::strtok(NULL, sep);
    }
    DEBUG_PRINT("device ddk version is  %s\n", sep_dev);
    cmp_times = 0;
    res_bin = std::strtok(bin_ddk_string, sep);
    while (res_bin && cmp_times++ < 3) {
        std::strcat(sep_bin, res_bin);
        res_bin = std::strtok(NULL, sep);
    }
    DEBUG_PRINT("binary ddk version is  %s\n", sep_bin);

    int ddk_not_same = strcmp(sep_bin, sep_dev);
    if (ddk_not_same) {
        LOGW(EDEN_CL, "device ddk version is  %s\n", sep_dev);
        LOGW(EDEN_CL, "binary ddk version is  %s\n", sep_bin);
        LOGW(EDEN_CL, "binary version is dismatched from device driver version\n");
    }
    delete[] sep_bin;
    delete[] sep_dev;
    delete[] bin_ddk_string;
    delete[] driver_string;
    if (ddk_not_same) {
        ERROR_PRINT_RETURN_FAILURE("binary version is dismatched from device driver version\n");
    }
    DEBUG_PRINT("binary kernel check passed\n");

    // saved binary crc
    int rd_binary_crc = 0;
    kernel_binary.read((char *)&rd_binary_crc, 4);

    // read and create kernel
    char *bin_kernel_string = new char[bin_ker_lenth];
    kernel_binary.read(bin_kernel_string, bin_ker_lenth);

    // check generated binary crc
    int gen_binary_crc = 0;
    gen_binary_crc = crc16(bin_kernel_string, bin_ker_lenth, 0);
    if (rd_binary_crc != gen_binary_crc) {
        delete[] bin_kernel_string;
        LOGW(EDEN_CL, "Generated binary crc dismatched %d %d\n", rd_binary_crc, gen_binary_crc);
        ERROR_PRINT_RETURN_FAILURE(
            "Generated binary crc dismatched %d %d\n", rd_binary_crc, gen_binary_crc);
    }

    size_t const_bin_ker_lenth = bin_ker_lenth;
    DEBUG_PRINT("kernel bin size is %zd\n", const_bin_ker_lenth);
    program_ = clCreateProgramWithBinary(context_,
                                         1,
                                         selected_device_,
                                         &const_bin_ker_lenth,
                                         (const unsigned char **)(&bin_kernel_string),
                                         NULL,
                                         &err);
    delete[] bin_kernel_string;
    if (err != CL_SUCCESS) {
        clReleaseContext(context_);
        clReleaseCommandQueue(queue_);
        LOGE(EDEN_CL, "clCreateProgramWithBinary() fail: %d\n", err);
        ERROR_PRINT_RETURN_CL_FAILURE("clCreateProgramWithBinary() fail: %d", err);
    }
    // build program
    const char options[] = "-cl-std=CL1.2 -cl-mad-enable";
    err = clBuildProgram(program_, 1, selected_device_, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        printProgramBuildInfo(program_);
        clReleaseContext(context_);
        clReleaseCommandQueue(queue_);
        clReleaseProgram(program_);
        LOGE(EDEN_CL, "clBuildProgram() fail\n");
        ERROR_PRINT_RETURN_CL_FAILURE("clBuildProgram() fail");
    }

    return Status::SUCCESS;
}

std::string CLRuntime::getKernelBinaryDir() {
    std::string dir_name;
#if defined(__ANDROID__)
    dir_name = "/data/vendor/eden/gpu";
#elif (__linux__)
    dir_name = "/lib/firmware";
#endif
    return dir_name;
}

std::string CLRuntime::getKernelBinaryName() {
    std::string kernel_binary_name;
#if defined(__ANDROID__)
    if (sizeof(long) == SYS64) {
        kernel_binary_name = "enn_kernel_64.bin";
    } else if (sizeof(long) == SYS32) {
        kernel_binary_name = "enn_kernel_32.bin";
    }
#elif (__linux__)
    kernel_binary_name = "enn_kernel_64.bin";
#endif
    return kernel_binary_name;
}

const std::string &CLRuntime::getKernelSourceByName(const std::string &kernel_name) {
    static std::string kernel;
    kernel = kernel_header_;
    if (Kernels::KernelHeaderMap().find(kernel_name) != Kernels::KernelHeaderMap().end()) {
        APPEND_PRIVATE_KERNEL_HEADERS(kernel, Kernels::KernelHeaderMap().at(kernel_name));
    }
    if (Kernels::KernelMap().find(kernel_name) != Kernels::KernelMap().end()) {
        kernel += Kernels::KernelMap().at(kernel_name);
        return kernel;
    }
    LOGE(EDEN_CL, "Kernel [%s] not Found!\n", kernel_name.c_str());
    kernel = "";
    return kernel;
}

Status CLRuntime::initializeProgramKernelSources() {
    DEBUG_PRINT("initializeProgramKernelSources() is called \n");
    LOGI(EDEN_CL, "==== Begin initializeProgramKernelSources ===\n");
    Status ret;
    ret = genOriginalKernelStringCRC();
    CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "CLRuntime::genOriginalKernelStringCRC() fail");
    LOGI(EDEN_CL, "==== End initializeProgramKernelSources ===\n");
    return ret;
}

Status CLRuntime::initializeProgram() {
    DEBUG_PRINT("initializeProgram() is called \n");
    Status ret;
    ret = genOriginalKernelStringCRC();
    CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "CLRuntime::genKernelString() fail");
    // this path should be changed to some place that enn service has file-write privilege,
    // in case of users delete the bin and cause regenerate work.
    // the kernel binary path has changed /data/vendor/eden/gpu -> /vendor/eden/gpu
    const std::string kernel_dir_name = getKernelBinaryDir();
    const std::string kernel_binary_name = getKernelBinaryName();
    const std::string file_name_optional = kernel_dir_name + "/" + kernel_binary_name;
    std::string file_name;
    std::string file_name_R;  // add path for android R
#if defined(__ANDROID__)
    file_name = std::string("/vendor/eden/gpu/") + kernel_binary_name;
    file_name_R = std::string("/vendor/etc/eden/gpu/") + kernel_binary_name;
#elif (__linux__)
    file_name = file_name_optional;
    file_name_R = file_name;
#endif

    LOGW(EDEN_CL, "Try to initialize first file -> : %s\n", file_name_R.c_str());
    Status binary_kernel_ok = tryInitializeProgramFromBinary(file_name_R);
    LOGW(EDEN_CL, "Finish initialize program from first binary file, ret = %d\n", binary_kernel_ok);
    if (binary_kernel_ok == Status::SUCCESS) {
        copyBinaryKernel(file_name_R, file_name_optional);
        return Status::SUCCESS;
    }

    LOGW(EDEN_CL, "Try to initialize second file -> : %s\n", file_name.c_str());
    binary_kernel_ok = tryInitializeProgramFromBinary(file_name);
    LOGW(EDEN_CL, "Finish initialize program from second binary file, ret = %d\n", binary_kernel_ok);
    if (binary_kernel_ok == Status::SUCCESS) {
        copyBinaryKernel(file_name, file_name_optional);
        return Status::SUCCESS;
    }

    LOGW(EDEN_CL, "Try to initialize third file -> : %s\n", file_name_optional.c_str());
    binary_kernel_ok = tryInitializeProgramFromBinary(file_name_optional);
    LOGW(EDEN_CL, "Finish initialize program from third binary file, ret = %d\n", binary_kernel_ok);
    if (binary_kernel_ok == Status::SUCCESS) {
        return Status::SUCCESS;
    }

#ifdef BENCHMARK
    // Search the file paths for initializing the gpu binary kernel for Benhcmark app
    // AITuTu and AIMark need other pathes that they can access avoiding 3rd party app's access issue
    // eden_kernel_64_r16p.bin is for DDK of OpenCL 2.0 v1.r16p0-01rel0
    // eden_kernel_64_r19p.bin is for DDK of OpenCL 2.1 v1.r19p0-01rel0
    // eden_kernel_64_r20p.bin is for DDK of OpenCL 2.1 v1.r20p0-01rel0
    std::vector<std::string> filePathsForBenchmark{
        "/sdcard/Samsung/Eden/eden_kernel_64.bin",
        "/sdcard/Samsung/Eden/eden_kernel_64_r16p.bin",
        "/sdcard/Samsung/Eden/eden_kernel_64_r19p.bin",
        "/sdcard/Samsung/Eden/eden_kernel_64_r20p.bin",
        "/sdcard/Android/data/com.antutu.aibenchmark/files/aitutupack/eden_kernel_64.bin",
        "/sdcard/Android/data/com.antutu.aibenchmark/files/aitutupack/eden_kernel_64_r16p.bin",
        "/sdcard/Android/data/com.antutu.aibenchmark/files/aitutupack/eden_kernel_64_r19p.bin",
        "/sdcard/Android/data/com.antutu.aibenchmark/files/aitutupack/eden_kernel_64_r20p.bin",
        "/storage/emulated/0/Android/data/com.ludashi.aibench/files/ai_data/samsung/eden_kernel_64.bin",
        "/storage/emulated/0/Android/data/com.ludashi.aibench/files/ai_data/samsung/eden_kernel_64_r16p.bin",
        "/storage/emulated/0/Android/data/com.ludashi.aibench/files/ai_data/samsung/eden_kernel_64_r19p.bin",
        "/storage/emulated/0/Android/data/com.ludashi.aibench/files/ai_data/samsung/eden_kernel_64_r20p.bin"};
    for (auto filePathForBenchmark : filePathsForBenchmark) {
        LOGW(EDEN_CL, "Try to initialize file for Benchmark -> : %s", filePathForBenchmark.c_str());
        if (tryInitializeProgramFromBinary(filePathForBenchmark) == Status::SUCCESS) {
            return Status::SUCCESS;
        }
    }
#endif

    {
        LOGW(EDEN_CL, "Try to generate gpu kernel binary -> %s\n", file_name_optional.c_str());
        // Need to check that this process can(or not) access and write to second file location
        binary_kernel_ok = checkAccessiblePath(file_name_optional);
        if (binary_kernel_ok != Status::SUCCESS) {
            LOGW(EDEN_CL,
                 "Try to generate gpu kernel binary failed as no access to path: %s."
                 " Try online_compile...\n",
                 file_name_optional.c_str());
            is_online_compile_ = true;
            ret = initializeProgramKernelSources();
            LOGW(EDEN_CL, "Finish initialize program from online_compile, ret = %d\n", ret);
            return ret;
        }
        LOGW(EDEN_CL, "Initialize program from kernel binary file failed, try initialize from source...\n");

        ret = initializeProgramFromSource();
        LOGW(EDEN_CL, "Finish initialize program from source, ret = %d\n", ret);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "CLRuntime::initializeProgramFromSource() fail");
        LOGW(EDEN_CL, "Try generate new kernel binary file, kernel_file_path: %s\n", file_name_optional.c_str());
        ret = generateBinaryKernel(kernel_dir_name, file_name_optional);
        LOGW(EDEN_CL, "Finish generate new kernel binary file, ret = %d\n", ret);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "CLRuntime::generateBinaryKernel() fail");

        // release resource and re-create
        // release opencl kernel instance
        for (auto &iter : kernels_) {
            iter.second = nullptr;
        }
        kernels_.clear();
        // release opencl kernel string
        std::vector<std::string> tmp;
        final_kernel_strings_.clear();
        final_kernel_strings_.swap(tmp);
        // release opencl instance
        clReleaseProgram(program_);
        program_ = nullptr;
        // re-create opencl instance
        LOGW(EDEN_CL, "Try to initialize program from the new kernel binary file -> : %s\n",
             file_name_optional.c_str());
        binary_kernel_ok = tryInitializeProgramFromBinary(file_name_optional);
        LOGW(EDEN_CL, "Finish initialize program from the new binary file, ret = %d\n", binary_kernel_ok);
        if (Status::SUCCESS == binary_kernel_ok) {
            return Status::SUCCESS;
        } else {
            LOGW(EDEN_CL, "Initialize program from the new binary file failed. Last struggle, try online_compile...\n");
            is_online_compile_ = true;
            ret = initializeProgramKernelSources();
            LOGW(EDEN_CL, "Finish initialize program from online_compile, ret = %d\n", ret);
            return ret;
        }
    }

    return Status::SUCCESS;
}

Status CLRuntime::initializeQueue() {
    if (queue_ == nullptr) {
        cl_int err = 0;
        cl_command_queue_properties queue_properties = 0;
        queue_ = clCreateCommandQueue(context_, *selected_device_, queue_properties, &err);
        CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clCreateCommandQueue() error, err: %d\n", err);
    }
    return Status::SUCCESS;
}

Status CLRuntime::initialize(const uint32_t &target_device_id) {
    DEBUG_PRINT("CLRuntime::initialize() is called");
    is_online_compile_ = true;
#ifdef EDEN_ONLINE_COMPILE_KERNEL
    is_online_compile_ = true;
#endif
    Status ret;
    platform_ = CLPlatform::getInstance();
    ret = platform_->initialize();
    CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "CLRuntime:: platform initialize fail");

    ret = initializeDevice(target_device_id);
    CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "CLRuntime::initializeDevice() fail");

    ret = initializeContext();
    CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "CLRuntime::initializeContext() fail");
    LOGI(EDEN_CL, "is_online_compile_ = %d\n", is_online_compile_);

    if (is_online_compile_) {
        ret = initializeProgramKernelSources();
        LOGI(EDEN_CL, "Finish initializeProgramKernelSources, ret = %d\n", ret);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "CLRuntime::initializeProgramKernelSources() fail");
    } else {
        ret = initializeProgram();
        LOGI(EDEN_CL, "Finish initializeProgram, ret = %d\n", ret);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "CLRuntime::initializeProgram() fail");
    }

    // ret = initializeQueue(); // temprary solution for DLV3 SW overhead increased issue
    // CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "CLRuntime::initializeQueue() fail");

    // pre-compile kernels for mcd model
    // CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == preCompileKernels(), "Failed to preCompileKernels.");
    return Status::SUCCESS;
}

Status CLRuntime::release() {
    DEBUG_PRINT("CLRuntime::release() is called");
    clReleaseProgram(program_);
    for (auto iter : programs_) {
        clReleaseProgram(iter);
    }
    clReleaseCommandQueue(queue_);
    clReleaseContext(context_);
    clReleaseDevice(*selected_device_);
    return Status::SUCCESS;
}

cl_device_id CLRuntime::getDeviceID(void) { return *selected_device_; }

bool CLRuntime::isFP16Support(void) { return is_fp16_support_; }

bool CLRuntime::isBifrost(void) { return is_bifrost_support_; }

bool CLRuntime::isMakalu(void) { return is_makalu_support_; }

bool CLRuntime::isValhall(void) { return is_valhall_support_; }

bool CLRuntime::isAMD(void) { return is_amd_; }

bool CLRuntime::isArmDotAccSupport(void) { return is_arm_dot_acc_support_; }

bool CLRuntime::isArmDotSupport(void) {return is_arm_dot_support_;}

cl_context CLRuntime::getContext(void) { return context_; }

cl_command_queue CLRuntime::getQueue(void) { return queue_; }

cl_program CLRuntime::getProgram(void) { return program_; }

Status CLRuntime::setKernelByName(std::shared_ptr<struct _cl_kernel> *kernel, const std::string &kernel_name) {

    auto search = kernels_.find(kernel_name);
    if (search != kernels_.end()) {
        *kernel = search->second;
    } else if (is_online_compile_) {
        const auto &kernel_str = getKernelSourceByName(kernel_name);
        if (kernel_str.empty()) {
            LOGI(EDEN_CL,
                 "clCreateKernel() fail: Not find kernel[%s] from final_kernel_strings_\n",
                 kernel_name.c_str());
            return Status::CL_FAILURE;
        }
        cl_int err;
        const char *final_char_source = kernel_str.c_str();
        cl_program program = clCreateProgramWithSource(context_, 1, &final_char_source, NULL, &err);
        if (err != CL_SUCCESS) {
            ERROR_PRINT_RETURN_FAILURE("clCreateProgramWithSource() fail: %d (%s)", err, kernel_name.c_str());
        }
        constexpr const char options[] = "-cl-std=CL1.2 -cl-mad-enable";
        err = clBuildProgram(program, 1, selected_device_, options, NULL, NULL);
        if (err != CL_SUCCESS) {
            std::cout << "[" << kernel_name << "] Build log:" << std::endl;
            printProgramBuildInfo(program);
            clReleaseProgram(program);
            LOGE(EDEN_CL, "clBuildProgram() fail: %d (%s)", err, kernel_name.c_str());
            return Status::CL_FAILURE;
        }
        cl_kernel opencl_kernel = clCreateKernel(program, kernel_name.c_str(), &err);
        CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clCreateKernel() fail: %d (%s)", err, kernel_name.c_str());
        auto deleter = [](cl_kernel kernel) { clReleaseKernel(kernel); };
        kernel->reset(opencl_kernel, deleter);
        kernels_[kernel_name] = *kernel;
        programs_.push_back(program);
    } else {
        cl_int err;
        cl_kernel opencl_kernel = clCreateKernel(program_, kernel_name.c_str(), &err);
        CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clCreateKernel() fail: %d (%s)", err, kernel_name.c_str());
        auto deleter = [](cl_kernel kernel) {
            clReleaseKernel(kernel);
        };
        kernel->reset(opencl_kernel, deleter);
        kernels_[kernel_name] = *kernel;
    }
    return Status::SUCCESS;
}

Status CLRuntime::setKernel(std::shared_ptr<_cl_kernel> *kernel,
                            const std::string &name,
                            const PrecisionType &type) {
    ENN_DBG_PRINT("setKernel %s\n", name.c_str());
    std::string kernel_name;
    if (type == PrecisionType::FP16) {
        kernel_name = name + "_FP16";
    } else if (type == PrecisionType::FP32) {
        kernel_name = name + "_FP32";
    } else {
        kernel_name = name + "_INT8";
    }
    return setKernelByName(kernel, kernel_name);
}

Status CLRuntime::preCompileKernels() {
    const char *common_kernels[] = {"align_weight_4_row_1_col_FP16",
                                    "align_weight_direct_FP16",
                                    "avepooling_caffe_FP16",
                                    "col2img_1x8_opt_FP16",
                                    "concat_axis1_size6_FP16",
                                    "concat_axis2_FP16",
                                    "convert_input_4_thread_8_1x1_FP16",
                                    "convert_input_4_thread_8_withpad_FP16",
                                    "copy_buffer_FP16",
                                    "depthwise_conv_3x3s1_pad_merge_FP16",
                                    "depthwise_conv_3x3s2_pad_merge_FP16",
                                    "depthwise_conv_FP16",
                                    "direct3x3_8x8_FP16",
                                    "direct3x3_4x4x2_FP16",
                                    "eltwise_add_zero_one_FP16",
                                    "eltwise_mul_zero_one_FP16",
                                    "Fast3x3_1_4x4_FP16",
                                    "Fast3x3_2_optimized_4x4_merge_FP16",
                                    "Fast3x3_2_optimized_4x4_merge_unaligned_FP16",
                                    "Fast3x3_2_optimized_4x4_splite_FP16",
                                    "float2half_FP16",
                                    "gemm_makalu_deconv_opt_FP16",
                                    "half2float_FP16",
                                    "maxpooling_FP16",
                                    "normalization_uint8_FP16",
                                    "pad4_FP16",
                                    "pure_matrix_transpose_FP16",
                                    "relu_FP16",
                                    "RELUcol2img_1x8_opt_FP16",
                                    "RELUeltwise_add_zero_one_FP16",
                                    "RELUFast3x3_2_optimized_4x4_merge_FP16",
                                    "reshape_FP16",
                                    "scale_FP16",
                                    "sigmoid_FP16",
                                    "tanh_FP16",
                                    "wino_convert_weight_FP16",
                                    "wino_convert_weight_makalu_opt_FP16",
                                    "zeroBuf_FP16"};

    std::shared_ptr<struct _cl_kernel> kernel_handle;
    for (size_t i = 0; i < sizeof(common_kernels) / sizeof(char *); ++i) {
        Status ret = setKernelByName(&kernel_handle, common_kernels[i]);
        if (ret != Status::SUCCESS) {
            return ret;
        }
    }

    if (is_valhall_support_) {
        const char *valhall_kernels[] = {"gemmValhall4x4_FP16", "RELUgemmValhall4x4_FP16"};
        for (size_t i = 0; i < sizeof(valhall_kernels) / sizeof(char *); ++i) {
            Status ret = setKernelByName(&kernel_handle, valhall_kernels[i]);
            if (ret != Status::SUCCESS) {
                return ret;
            }
        }
    } else if (is_makalu_support_) {
        const char *makalu_kernels[] = {"gemmMakalu_FP16", "RELUgemmMakalu_FP16"};
        for (size_t i = 0; i < sizeof(makalu_kernels) / sizeof(char *); ++i) {
            Status ret = setKernelByName(&kernel_handle, makalu_kernels[i]);
            if (ret != Status::SUCCESS) {
                return ret;
            }
        }
    }
    return Status::SUCCESS;
}

Status CLRuntime::setCommonKernels() {
    Status ret = Status::SUCCESS;
    if (kernel_float2half_ == nullptr) {
        ret = setKernel(&kernel_float2half_, "float2half", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == ret, "setKernel failure\n");
    }
    if (kernel_half2float_ == nullptr) {
        ret = setKernel(&kernel_half2float_, "half2float", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == ret, "setKernel failure\n");
    }
    return ret;
}

Status CLRuntime::setPublicBufferKernels() {
    Status ret = Status::SUCCESS;
    if (broadcast_kernel_ == nullptr) {
        ret = setKernel(&broadcast_kernel_, "broadcast", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "CLTensor::broadCast() setKernel fail");
        ret = setKernel(&broadcast_kernel_, "broadcast", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "CLTensor::broadCast() setKernel fail");
    }
    return ret;
}

Status CLRuntime::setPublicTexture2dKernels() {
    Status ret = Status::SUCCESS;
    if (kernel_half2float_texture2d_ == nullptr) {
        ret = setKernel(&kernel_half2float_texture2d_, "half2float_texture2d", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == ret, "setKernel failure\n");
    }
    if (kernel_float2half_texture2d_ == nullptr) {
        ret = setKernel(&kernel_float2half_texture2d_, "float2half_texture2d", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == ret, "setKernel failure\n");
    }
    if (kernel_NHWC2DHWC4_fp322fp16_ == nullptr) {
        ret = setKernel(&kernel_NHWC2DHWC4_fp322fp16_, "nhwc2dhwc4_fp322fp16", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_NHWC2DHWC4_fp322fp16 setKernel fail");
    }
    if (kernel_NHWC2DHWC4_fp162fp32_ == nullptr) {
        ret = setKernel(&kernel_NHWC2DHWC4_fp162fp32_, "nhwc2dhwc4_fp162fp32", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_NHWC2DHWC4_fp162fp32 setKernel fail");
    }
    if (kernel_NHWC2DHWC4_fp32_ == nullptr) {
        ret = setKernel(&kernel_NHWC2DHWC4_fp32_, "nhwc2dhwc4", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_NHWC2DHWC4_fp32 setKernel fail");
    }
    if (kernel_NHWC2DHWC4_fp16_ == nullptr) {
        ret = setKernel(&kernel_NHWC2DHWC4_fp16_, "nhwc2dhwc4", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_NHWC2DHWC4_fp16 setKernel fail");
    }
    if (kernel_NCHW2DHWC4_fp322fp16_ == nullptr) {
        ret = setKernel(&kernel_NCHW2DHWC4_fp322fp16_, "nchw2dhwc4_fp322fp16", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_NCHW2DHWC4_fp322fp16 setKernel fail");
    }
    if (kernel_NCHW2DHWC4_fp162fp32_ == nullptr) {
        ret = setKernel(&kernel_NCHW2DHWC4_fp162fp32_, "nchw2dhwc4_fp162fp32", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_NCHW2DHWC4_fp162fp32 setKernel fail");
    }
    if (kernel_NCHW2DHWC4_fp32_ == nullptr) {
        ret = setKernel(&kernel_NCHW2DHWC4_fp32_, "nchw2dhwc4", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_NCHW2DHWC4_fp32 setKernel fail");
    }
    if (kernel_NCHW2DHWC4_fp16_ == nullptr) {
        ret = setKernel(&kernel_NCHW2DHWC4_fp16_, "nchw2dhwc4", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_NCHW2DHWC4_fp16 setKernel fail");
    }
    if (kernel_DHWC42NHWC_fp162fp32_ == nullptr) {
        ret = setKernel(&kernel_DHWC42NHWC_fp162fp32_, "dhwc42nhwc_fp162fp32", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_DHWC42NHWC_fp162fp32 setKernel fail");
    }
    if (kernel_DHWC42NHWC_fp322fp16_ == nullptr) {
        ret = setKernel(&kernel_DHWC42NHWC_fp322fp16_, "dhwc42nhwc_fp322fp16", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_DHWC42NHWC_fp322fp16 setKernel fail");
    }
    if (kernel_DHWC42NHWC_fp16_ == nullptr) {
        ret = setKernel(&kernel_DHWC42NHWC_fp16_, "dhwc42nhwc", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_DHWC42NCHW_fp16 setKernel fail");
    }
    if (kernel_DHWC42NHWC_fp32_ == nullptr) {
        ret = setKernel(&kernel_DHWC42NHWC_fp32_, "dhwc42nhwc", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_DHWC42NCHW_fp32 setKernel fail");
    }
    if (kernel_DHWC42NCHW_fp162fp32_ == nullptr) {
        ret = setKernel(&kernel_DHWC42NCHW_fp162fp32_, "dhwc42nchw_fp162fp32", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_DHWC42NCHW_fp162fp32 setKernel fail");
    }
    if (kernel_DHWC42NCHW_fp322fp16_ == nullptr) {
        ret = setKernel(&kernel_DHWC42NCHW_fp322fp16_, "dhwc42nchw_fp322fp16", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_DHWC42NCHW_fp322fp16 setKernel fail");
    }
    if (kernel_DHWC42NCHW_fp16_ == nullptr) {
        ret = setKernel(&kernel_DHWC42NCHW_fp16_, "dhwc42nchw", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_DHWC42NCHW_fp16 setKernel fail");
    }
    if (kernel_DHWC42NCHW_fp32_ == nullptr) {
        ret = setKernel(&kernel_DHWC42NCHW_fp32_, "dhwc42nchw", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "kernel_DHWC42NCHW_fp32 setKernel fail");
    }
    return ret;
}

Status CLRuntime::zeroBuf(const int &size, cl_mem buf) {
    DEBUG_PRINT("CLRuntime::zeroBuf() is called");
    std::shared_ptr<_cl_kernel> zero_buf_ = nullptr;
    Status status = setKernel(&zero_buf_, "zeroBuf", PrecisionType::FP16);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");

    const size_t global = alignTo(ceil(size / sizeof(cl_half) / 8.0), 8);
    const size_t local = 8;
    int halfSize = size / sizeof(cl_half);
    status = this->setKernelArg(zero_buf_.get(), buf, halfSize);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

    status = this->enqueueKernel(zero_buf_.get(), (cl_uint)1, &global, &local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");
    return Status::SUCCESS;
}

Status CLRuntime::zeroTexture2D(const TextureDescriptor &texture_descriptor,
                                cl_mem buf) {
    DEBUG_PRINT("CLRuntime::zeroTexture2D() is called");

    std::shared_ptr<_cl_kernel> zero_texture2d_ = nullptr;
    Status status = setKernel(&zero_texture2d_, "zeroTexture2D", texture_descriptor.precision);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");

    status = setKernelArg(zero_texture2d_.get(), buf, texture_descriptor.image_width, texture_descriptor.image_height);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t local[2] = {16, 4};
    size_t global[2] = {0};
    global[0] = alignTo(texture_descriptor.image_width, local[0]);
    global[1] = alignTo(texture_descriptor.image_height, local[1]);

    status = enqueueKernel(zero_texture2d_.get(), (cl_uint)2, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");

    return Status::SUCCESS;
}

cl_channel_type CLRuntime::ToImageChannelType(PrecisionType precision) {
    switch (precision) {
        case PrecisionType::FP32:
            return CL_FLOAT;
        case PrecisionType::FP16:
            return CL_HALF_FLOAT;
        default:
            return -1;
    }
}

cl_mem CLRuntime::CreateImage2DLegacy(cl_context context,
                                      cl_mem_flags flags,
                                      const cl_image_format *image_format,
                                      const cl_image_desc *image_desc,
                                      void *host_ptr,
                                      cl_int *errcode_ret) {
    if (1) {  // clCreateImage) {  // clCreateImage available since OpenCL 1.2
        return clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret);
    } else {
        return clCreateImage2D(context,
                               flags,
                               image_format,
                               image_desc->image_width,
                               image_desc->image_height,
                               image_desc->image_row_pitch,
                               host_ptr,
                               errcode_ret);
    }
}

cl_mem CLRuntime::allocTexture2D(const TextureDescriptor &texture_descriptor,
                                 const bool &zero_init) {
    cl_image_desc desc;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = texture_descriptor.image_width;
    desc.image_height = texture_descriptor.image_height;
    desc.image_depth = 0;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.buffer = nullptr;

    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = ToImageChannelType(texture_descriptor.precision);

    cl_int error_code = CL_SUCCESS;
    cl_mem memory =
        CreateImage2DLegacy(context_, CL_MEM_READ_WRITE, &format, &desc, nullptr, &error_code);
    CHECK_EXPR_RETURN_NULL(CL_SUCCESS == error_code, "Error: CreateImage2DLegacy error, err: %d", error_code);
    if (zero_init) {
        Status ret = zeroTexture2D(texture_descriptor, memory);
        CHECK_EXPR_RETURN_NULL(Status::SUCCESS == ret, "set zero err in malloc");
    }
    return memory;
}

cl_mem CLRuntime::allocBuffer(const uint32_t &bytes, const bool &zero_init, DataPtr data) {
    ENN_UNUSED(zero_init);
    cl_mem buffer = nullptr;
    cl_int err = CL_SUCCESS;

    buffer = clCreateBuffer(
        context_, CL_MEM_USE_HOST_PTR, bytes, (void *)data, &err);  // creat buffer by host_ptr
    CHECK_EXPR_RETURN_NULL(CL_SUCCESS == err, "clCreateBuffer() fail: %d\n", err);
    return buffer;
}

cl_mem CLRuntime::allocBuffer(const uint32_t &bytes, const bool &zero_init) {
    cl_mem buffer = nullptr;
    cl_mem subBuffer = nullptr;
    cl_int err = CL_SUCCESS;
    if (bytes != 0) {
        // pad 1024 elements to avoid the memory overread since vector instructions are used for
        // optimization
        size_t alignHead = 1024;
        size_t alignTail =
            32;  // sizeof(cl_float) * 8, double precision need to be considered in the future.
        size_t alignSize = 1024;
        size_t bufSize = alignTo(alignHead + alignTail + bytes, alignSize);
        size_t subBufSize = bytes;
        buffer = clCreateBuffer(
            context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufSize, NULL, &err);
        CHECK_EXPR_RETURN_NULL(CL_SUCCESS == err, "clCreateBuffer() fail: %d\n", err);

        cl_buffer_region bufRegin;
        bufRegin.origin = alignHead;
        bufRegin.size = subBufSize;
        subBuffer = clCreateSubBuffer(
            buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bufRegin, &err);
        CHECK_EXPR_RETURN_NULL(CL_SUCCESS == err, "Error: clCreateSubBuf error, err: %d", err);
        if (zero_init) {
            Status ret = zeroBuf(bufSize, buffer);
            CHECK_EXPR_RETURN_NULL(Status::SUCCESS == ret, "set zero err in malloc");
        }
        clReleaseMemObject(buffer);
        DEBUG_PRINT("CLRuntime::allocBuffer() %p", buffer);
    } else {
        DEBUG_PRINT("CLRuntime::allocBuffer() bytes = %d", bytes);
    }
    return subBuffer;
}

Status CLRuntime::writeBuffer(cl_mem dst,
                              void *src,
                              size_t type_bytes,
                              uint32_t num,
                              cl_bool blocking) {
    cl_int err = CL_SUCCESS;
    std::lock_guard<std::mutex> lock(mutex_);
    DEBUG_PRINT("CLRuntime::writeBuffer() is called");

    err = clEnqueueWriteBuffer(queue_, dst, blocking, 0, type_bytes * (size_t)num, src, 0, NULL, NULL);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clEnqueueWriteBuffer() fail (%d)", err);
    return Status::SUCCESS;
}

Status CLRuntime::readBuffer(void *dst,
                             cl_mem src,
                             size_t type_bytes,
                             uint32_t num,
                             cl_bool blocking,
                             void *evt) {
    cl_int err = CL_SUCCESS;

    std::lock_guard<std::mutex> lock(mutex_);
    DEBUG_PRINT("CLRuntime::readBuffer()  is called");
    if (blocking && evt != nullptr) {
        cl_event *event = static_cast<cl_event *>(evt);
        err = clEnqueueReadBuffer(queue_, src, false, 0, type_bytes * (size_t)num, dst, 0, NULL, event);
        clFlush(queue_);
    } else {
        err = clEnqueueReadBuffer(queue_, src, blocking, 0, type_bytes * (size_t)num, dst, 0, NULL, NULL);
    }
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clEnqueueReadBuffer() fail %d", err);
    return Status::SUCCESS;
}

Status CLRuntime::releaseBuffer(std::shared_ptr<CLBuffer> buffer) {
    std::lock_guard<std::mutex> lock(mutex_);

    cl_int err = clReleaseMemObject(buffer->getDataPtr());
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "CLRuntime::releaseBuffer() fail");
    buffer->assignBuffer(nullptr);
    return Status::SUCCESS;
}

Status CLRuntime::copyFloat2Half(cl_mem dst, cl_mem src, const uint32_t &num) {
    Status status = Status::SUCCESS;
    if (kernel_float2half_ == nullptr) {
        status = setKernel(&kernel_float2half_, "float2half", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");
    }

    status = setKernelArg(kernel_float2half_.get(), src, dst);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t global = num;
    status = enqueueKernel(kernel_float2half_.get(), (cl_uint)1, &global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");
    return Status::SUCCESS;
}

Status CLRuntime::copyHalf2Float(cl_mem dst, cl_mem src, const uint32_t &num) {
    Status status = Status::SUCCESS;
    if (kernel_half2float_ == nullptr) {
        status = setKernel(&kernel_half2float_, "half2float", PrecisionType::FP16);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");
    }

    status = setKernelArg(kernel_half2float_.get(), src, dst);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t global = num;
    status = enqueueKernel(kernel_half2float_.get(), (cl_uint)1, &global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");
    return Status::SUCCESS;
}

Status CLRuntime::copyInt2Float(cl_mem dst, cl_mem src, const uint32_t &num) {
    Status status = Status::SUCCESS;
    if (kernel_int2float_ == nullptr) {
        status = setKernel(&kernel_int2float_, "int2float", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");
    }

    status = setKernelArg(kernel_int2float_.get(), src, dst);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t global = num;
    status = enqueueKernel(kernel_int2float_.get(), (cl_uint)1, &global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");
    return Status::SUCCESS;
}

Status CLRuntime::copyFloat2Int(cl_mem dst, cl_mem src, const uint32_t &num) {
    Status status = Status::SUCCESS;
    if (kernel_float2int_ == nullptr) {
        status = setKernel(&kernel_float2int_, "float2int", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");
    }
    status = setKernelArg(kernel_float2int_.get(), src, dst);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t global = num;
    status = enqueueKernel(kernel_float2int_.get(), (cl_uint)1, &global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");
    return Status::SUCCESS;
}

Status CLRuntime::NCHW2NHWC(cl_mem src,
                            cl_mem dst,
                            const Dim4 &dim,
                            DataType type,
                            PrecisionChangeMode mode) {
    std::shared_ptr<_cl_kernel> kernel_NCHW2NHWC;
    Status status = Status::SUCCESS;
    if (mode == PrecisionChangeMode::FP32_TO_FP16) {
        if (kernel_NCHW2NHWC_fp322fp16_ == nullptr) {
            status = setKernel(&kernel_NCHW2NHWC_fp322fp16_, "nchw2nhwc_fp32", PrecisionType::FP16);
            CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NCHW2NHWC_fp322fp16_ setKernel fail");
        }
        kernel_NCHW2NHWC = kernel_NCHW2NHWC_fp322fp16_;
    } else if (mode == PrecisionChangeMode::FP16_TO_FP32) {
        if (kernel_NCHW2NHWC_fp162fp32_ == nullptr) {
            status = setKernel(&kernel_NCHW2NHWC_fp162fp32_, "nchw2nhwc_fp16", PrecisionType::FP32);
            CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NCHW2NHWC_fp162fp32_ setKernel fail");
        }
        kernel_NCHW2NHWC = kernel_NCHW2NHWC_fp162fp32_;

    } else {
        if (type == DataType::HALF) {
            if (kernel_NCHW2NHWC_fp16_ == nullptr) {
                status = setKernel(&kernel_NCHW2NHWC_fp16_, "nchw2nhwc", PrecisionType::FP16);
                CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NCHW2NHWC_fp16_ setKernel fail");
            }
            kernel_NCHW2NHWC = kernel_NCHW2NHWC_fp16_;
        } else if (type == DataType::FLOAT || type == DataType::INT32) {
            if (kernel_NCHW2NHWC_fp32_ == nullptr) {
                status = setKernel(&kernel_NCHW2NHWC_fp32_, "nchw2nhwc", PrecisionType::FP32);
                CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NCHW2NHWC_fp32_ setKernel fail");
            }
            kernel_NCHW2NHWC = kernel_NCHW2NHWC_fp32_;
        } else if (type == DataType::UINT8) {
            if (kernel_NCHW2NHWC_uint8_ == nullptr) {
                status = setKernel(&kernel_NCHW2NHWC_uint8_, "nchw2nhwc_uint8", PrecisionType::UINT8);
                CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NCHW2NHWC_uint8_ setKernel fail");
            }
            kernel_NCHW2NHWC = kernel_NCHW2NHWC_uint8_;
        } else if (type == DataType::INT8 || type == DataType::BOOL) {
            if (kernel_NCHW2NHWC_int8_ == nullptr) {
                status = setKernel(&kernel_NCHW2NHWC_int8_, "nchw2nhwc_int8", PrecisionType::UINT8);
                CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NCHW2NHWC_int8_ setKernel fail");
            }
            kernel_NCHW2NHWC = kernel_NCHW2NHWC_int8_;
        } else if (type == DataType::UINT16) {
            if (kernel_NCHW2NHWC_uint16_ == nullptr) {
                status = setKernel(&kernel_NCHW2NHWC_uint16_, "nchw2nhwc_uint16", PrecisionType::FP32);
                CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NCHW2NHWC_uint16_ setKernel fail");
            }
            kernel_NCHW2NHWC = kernel_NCHW2NHWC_uint16_;
        } else if (type == DataType::INT16) {
            if (kernel_NCHW2NHWC_int16_ == nullptr) {
                status = setKernel(&kernel_NCHW2NHWC_int16_, "nchw2nhwc_int16", PrecisionType::FP32);
                CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NCHW2NHWC_int16_ setKernel fail");
            }
            kernel_NCHW2NHWC = kernel_NCHW2NHWC_int16_;
        }
    }

    status = setKernelArg(kernel_NCHW2NHWC.get(), src, dst, dim.c, dim.h * dim.w);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t local[3] = {1, 1, 24};
    size_t global[3] = {0};
    global[0] = dim.n;
    global[1] = dim.c;
    global[2] = alignTo(dim.h * dim.w, local[2]);
    status = enqueueKernel(kernel_NCHW2NHWC.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");

    return Status::SUCCESS;
}

Status CLRuntime::NHWC2NCHW(cl_mem src,
                            cl_mem dst,
                            const Dim4 &dim,
                            DataType type,
                            PrecisionChangeMode mode) {
    std::shared_ptr<_cl_kernel> kernel_NHWC2NCHW;
    Status status = Status::SUCCESS;
    if (mode == PrecisionChangeMode::FP32_TO_FP16) {
        if (kernel_NHWC2NCHW_fp322fp16_ == nullptr) {
            status = setKernel(&kernel_NHWC2NCHW_fp322fp16_, "nhwc2nchw_fp32", PrecisionType::FP16);
            CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NHWC2NCHW_fp322fp16_ setKernel fail");
        }
        kernel_NHWC2NCHW = kernel_NHWC2NCHW_fp322fp16_;
    } else if (mode == PrecisionChangeMode::FP16_TO_FP32) {
        if (kernel_NHWC2NCHW_fp162fp32_ == nullptr) {
            status = setKernel(&kernel_NHWC2NCHW_fp162fp32_, "nhwc2nchw_fp16", PrecisionType::FP32);
            CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NHWC2NCHW_fp162fp32_ setKernel fail");
        }
        kernel_NHWC2NCHW = kernel_NHWC2NCHW_fp162fp32_;
    } else {
        if (type == DataType::HALF) {
            if (kernel_NHWC2NCHW_fp16_ == nullptr) {
                status = setKernel(&kernel_NHWC2NCHW_fp16_, "nhwc2nchw", PrecisionType::FP16);
                CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NHWC2NCHW_fp16_ setKernel fail");
            }
            kernel_NHWC2NCHW = kernel_NHWC2NCHW_fp16_;
        } else if (type == DataType::FLOAT || type == DataType::INT32) {
            if (kernel_NHWC2NCHW_fp32_ == nullptr) {
                status = setKernel(&kernel_NHWC2NCHW_fp32_, "nhwc2nchw", PrecisionType::FP32);
                CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NHWC2NCHW_fp32_ setKernel fail");
            }
            kernel_NHWC2NCHW = kernel_NHWC2NCHW_fp32_;
        } else if (type == DataType::UINT8) {
            if (kernel_NHWC2NCHW_uint8_ == nullptr) {
                status = setKernel(&kernel_NHWC2NCHW_uint8_, "nhwc2nchw_uint8", PrecisionType::UINT8);
                CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NHWC2NCHW_uint8_ setKernel fail");
            }
            kernel_NHWC2NCHW = kernel_NHWC2NCHW_uint8_;
        } else if (type == DataType::INT8 || type == DataType::BOOL) {
            if (kernel_NHWC2NCHW_int8_ == nullptr) {
                status = setKernel(&kernel_NHWC2NCHW_int8_, "nhwc2nchw_int8", PrecisionType::UINT8);
                CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NHWC2NCHW_int8_ setKernel fail");
            }
            kernel_NHWC2NCHW = kernel_NHWC2NCHW_int8_;
        } else if (type == DataType::UINT16) {
            if (kernel_NHWC2NCHW_uint16_ == nullptr) {
                status = setKernel(&kernel_NHWC2NCHW_uint16_, "nhwc2nchw_uint16", PrecisionType::FP32);
                CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NHWC2NCHW_uint16_ setKernel fail");
            }
            kernel_NHWC2NCHW = kernel_NHWC2NCHW_uint16_;
        } else if (type == DataType::INT16) {
            if (kernel_NHWC2NCHW_int16_ == nullptr) {
                status = setKernel(&kernel_NHWC2NCHW_int16_, "nhwc2nchw_int16", PrecisionType::FP32);
                CHECK_EXPR_RETURN_FAILURE(status == Status::SUCCESS, "kernel_NHWC2NCHW_int16_ setKernel fail");
            }
            kernel_NHWC2NCHW = kernel_NHWC2NCHW_int16_;
        }
    }
    status = setKernelArg(kernel_NHWC2NCHW.get(), src, dst, dim.c, dim.h * dim.w);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t local[3] = {1, 1, 24};
    size_t global[3] = {0};
    global[0] = dim.n;
    global[1] = dim.c;
    global[2] = alignTo(dim.h * dim.w, local[2]);
    status = enqueueKernel(kernel_NHWC2NCHW.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");

    return Status::SUCCESS;
}

Status CLRuntime::NHWC2DHWC4(cl_mem src,
                             cl_mem dst,
                             const Dim4 &dim,
                             DataType type,
                             PrecisionChangeMode mode) {
    std::shared_ptr<_cl_kernel> kernel_NHWC2DHWC4;
    uint32_t div = 4;
    int depth = IntegralDivideRoundUp(dim.c, div);
    if (mode == PrecisionChangeMode::FP32_TO_FP16) {
        kernel_NHWC2DHWC4 = kernel_NHWC2DHWC4_fp322fp16_;
    } else if (mode == PrecisionChangeMode::FP16_TO_FP32) {
        kernel_NHWC2DHWC4 = kernel_NHWC2DHWC4_fp162fp32_;
    } else {
        if (type == DataType::HALF) {
            kernel_NHWC2DHWC4 = kernel_NHWC2DHWC4_fp16_;
        } else if (type == DataType::FLOAT || type == DataType::INT32) {
            kernel_NHWC2DHWC4 = kernel_NHWC2DHWC4_fp32_;
        } else {
            LOGE(EDEN_CL, "data type is not supported");
            return Status::FAILURE;
        }
    }
    Status status =
        setKernelArg(kernel_NHWC2DHWC4.get(), src, dst, dim.n, dim.c, dim.h, dim.w, depth);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t local[3] = {8, 4, 4};
    size_t global[3] = {0};
    global[0] = alignTo(dim.w, local[0]);
    global[1] = alignTo(dim.h, local[1]);
    global[2] = alignTo(depth, local[2]);

    status = enqueueKernel(kernel_NHWC2DHWC4.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");

    return Status::SUCCESS;
}

Status CLRuntime::NCHW2DHWC4(cl_mem src,
                             cl_mem dst,
                             const Dim4 &dim,
                             DataType type,
                             PrecisionChangeMode mode) {
    std::shared_ptr<_cl_kernel> kernel_NCHW2DHWC4;
    uint32_t div = 4;
    int depth = IntegralDivideRoundUp(dim.c, div);
    if (mode == PrecisionChangeMode::FP32_TO_FP16) {
        kernel_NCHW2DHWC4 = kernel_NCHW2DHWC4_fp322fp16_;
    } else if (mode == PrecisionChangeMode::FP16_TO_FP32) {
        kernel_NCHW2DHWC4 = kernel_NCHW2DHWC4_fp162fp32_;
    } else {
        if (type == DataType::HALF) {
            kernel_NCHW2DHWC4 = kernel_NCHW2DHWC4_fp16_;
        } else if (type == DataType::FLOAT || type == DataType::INT32) {
            kernel_NCHW2DHWC4 = kernel_NCHW2DHWC4_fp32_;
        } else {
            LOGE(EDEN_CL, "data type is not supported");
            return Status::FAILURE;
        }
    }
    Status status =
        setKernelArg(kernel_NCHW2DHWC4.get(), src, dst, dim.n, dim.c, dim.h, dim.w, depth);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t local[3] = {8, 4, 4};
    size_t global[3] = {0};
    global[0] = alignTo(dim.w, local[0]);
    global[1] = alignTo(dim.h, local[1]);
    global[2] = alignTo(depth, local[2]);

    status = enqueueKernel(kernel_NCHW2DHWC4.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");
    return Status::SUCCESS;
}

Status CLRuntime::DHWC42NHWC(cl_mem src,
                             cl_mem dst,
                             const Dim4 &dim,
                             DataType type,
                             PrecisionChangeMode mode) {
    std::shared_ptr<_cl_kernel> kernel_DHWC42NHWC;
    uint32_t div = 4;
    int depth = IntegralDivideRoundUp(dim.c, div);
    if (mode == PrecisionChangeMode::FP32_TO_FP16) {
        kernel_DHWC42NHWC = kernel_DHWC42NHWC_fp322fp16_;
    } else if (mode == PrecisionChangeMode::FP16_TO_FP32) {
        kernel_DHWC42NHWC = kernel_DHWC42NHWC_fp162fp32_;
    } else {
        if (type == DataType::HALF) {
            kernel_DHWC42NHWC = kernel_DHWC42NHWC_fp16_;
        } else if (type == DataType::FLOAT || type == DataType::INT32) {
            kernel_DHWC42NHWC = kernel_DHWC42NHWC_fp32_;
        } else {
            LOGE(EDEN_CL, "data type is not supported");
            return Status::FAILURE;
        }
    }

    Status status =
        setKernelArg(kernel_DHWC42NHWC.get(), src, dst, dim.n, dim.c, dim.h, dim.w, depth);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t local[3] = {8, 4, 4};
    size_t global[3] = {0};
    global[0] = alignTo(dim.w, local[0]);
    global[1] = alignTo(dim.h, local[1]);
    global[2] = alignTo(depth, local[2]);
    status = enqueueKernel(kernel_DHWC42NHWC.get(), (cl_uint)3, global, local);
    return Status::SUCCESS;
}

Status CLRuntime::DHWC42NCHW(cl_mem src,
                             cl_mem dst,
                             const Dim4 &dim,
                             DataType type,
                             PrecisionChangeMode mode) {
    std::shared_ptr<_cl_kernel> kernel_DHWC42NCHW;
    uint32_t div = 4;
    int depth = IntegralDivideRoundUp(dim.c, div);
    if (mode == PrecisionChangeMode::FP32_TO_FP16) {
        kernel_DHWC42NCHW = kernel_DHWC42NCHW_fp322fp16_;
    } else if (mode == PrecisionChangeMode::FP16_TO_FP32) {
        kernel_DHWC42NCHW = kernel_DHWC42NCHW_fp162fp32_;
    } else {
        if (type == DataType::HALF) {
            kernel_DHWC42NCHW = kernel_DHWC42NCHW_fp16_;
        } else if (type == DataType::FLOAT || type == DataType::INT32) {
            kernel_DHWC42NCHW = kernel_DHWC42NCHW_fp32_;
        } else {
            LOGE(EDEN_CL, "data type is not supported");
            return Status::FAILURE;
        }
    }
    Status status =
        setKernelArg(kernel_DHWC42NCHW.get(), src, dst, dim.n, dim.c, dim.h, dim.w, depth);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t local[3] = {8, 4, 4};
    size_t global[3] = {0};
    global[0] = alignTo(dim.w, local[0]);
    global[1] = alignTo(dim.h, local[1]);
    global[2] = alignTo(depth, local[2]);

    status = enqueueKernel(kernel_DHWC42NCHW.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");

    return Status::SUCCESS;
}

Status CLRuntime::writeBufferTexture2D(cl_mem dst, void *src, Dim4 &dim, cl_bool blocking) {
    ENN_UNUSED(dst), ENN_UNUSED(src), ENN_UNUSED(dim), ENN_UNUSED(blocking);
    return Status::FAILURE;
}

Status CLRuntime::readBufferTexture2D(void *dst, cl_mem src, Dim4 &dim, cl_bool blocking) {
    std::lock_guard<std::mutex> lock(mutex_);
    DEBUG_PRINT("CLRuntime::readBufferTexture2D()  is called");

    const int slices = IntegralDivideRoundUp(dim.c, 4);
    const size_t origin[] = {0, 0, 0};
    const size_t r[] = {static_cast<size_t>(dim.w * dim.n),
                        static_cast<size_t>(dim.h * slices),
                        static_cast<size_t>(1)};
    cl_int err =
        clEnqueueReadImage(queue_, src, blocking, origin, r, 0, 0, dst, 0, nullptr, nullptr);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clEnqueueReadImage() fail %d", err);
    return Status::SUCCESS;
}

Status CLRuntime::copyFloat2HalfTexture2D(cl_mem dst, cl_mem src, Dim4 &dim) {
    const int slices = IntegralDivideRoundUp(dim.c, 4);
    int32_t image_width = dim.w * dim.n;
    int32_t image_height = dim.h * slices;

    Status status =
        setKernelArg(kernel_float2half_texture2d_.get(), src, dst, image_width, image_height);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t local[2] = {16, 4};
    size_t global[2] = {0};
    global[0] = alignTo(image_width, local[0]);
    global[1] = alignTo(image_height, local[1]);

    status = enqueueKernel(kernel_float2half_texture2d_.get(), (cl_uint)2, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");
    return Status::SUCCESS;
}
Status CLRuntime::copyHalf2FloatTexture2D(cl_mem dst, cl_mem src, Dim4 &dim) {
    const int slices = IntegralDivideRoundUp(dim.c, 4);
    int32_t image_width = dim.w * dim.n;
    int32_t image_height = dim.h * slices;

    Status status =
        setKernelArg(kernel_half2float_texture2d_.get(), src, dst, image_width, image_height);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t local[2] = {16, 4};
    size_t global[2] = {0};
    global[0] = alignTo(image_width, local[0]);
    global[1] = alignTo(image_height, local[1]);

    status = enqueueKernel(kernel_half2float_texture2d_.get(), (cl_uint)2, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");

    return Status::SUCCESS;
}
static uint64_t clflush_count = 0;
Status CLRuntime::enqueueKernel(const cl_kernel &kernel,
                                const cl_uint &work_dim,
                                const size_t *const &global_work_size,
                                const size_t *const &local_work_size) {
    cl_int err = clEnqueueNDRangeKernel(
        queue_, kernel, work_dim, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
    CHECK_EXPR_RETURN_FAILURE(
        CL_SUCCESS == err, "clEnqueueNDRangeKernel() (enqueue) fail: %d", err);
    const int32_t clflush_freq = is_amd_ ? 1 : 36;
    if (clflush_count++ % clflush_freq == 0) {
        clFlush(queue_);
    }
    return Status::SUCCESS;
}

Status CLRuntime::copyBuffer(cl_mem dst,
                             cl_mem src,
                             size_t dst_offset_bytes,
                             size_t src_offset_bytes,
                             size_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    cl_int err = clEnqueueCopyBuffer(
        queue_, src, dst, src_offset_bytes, dst_offset_bytes, size_bytes, 0, NULL, NULL);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "CLRuntime::copyBuffer() fail");
    return Status::SUCCESS;
}

std::vector<std::string> CLRuntime::removeSpecificKernel(std::vector<std::string> kernel_string,
                                                         const std::string &kernel_name) {
    std::string str;
    std::smatch match;
    std::regex name_re(kernel_name);
    for (auto iter = kernel_string.begin(); iter != kernel_string.end(); iter++) {
        str = *iter;
        if (std::regex_search(str, match, name_re)) {
            kernel_string.erase(iter);
            break;
        }
    }
    return kernel_string;
}

Status CLRuntime::preCreateOpenCLKernel(const std::vector<std::string> &opencl_kernels) {
    std::regex name_re("__kernel\\s+void\\s+(\\w+)\\(");
    int ikernel = 0;
    int nkernel = opencl_kernels.size();
    for (auto ocl_kernel : opencl_kernels) {
        if (ikernel++ % 16 == 0) {
            LOGI(EDEN_CL, "Progress: [%d / %d]", ikernel, nkernel);
        }
        std::smatch match;
        if (std::regex_search(ocl_kernel, match, name_re)) {
            std::shared_ptr<_cl_kernel> cl_shared_kernel;
            std::string kernel_name = match[match.size() - 1];

            auto search = kernels_.find(kernel_name);
            if (search != kernels_.end()) {
                continue;
            } else {
                cl_int err;
                cl_kernel opencl_kernel = clCreateKernel(program_, kernel_name.c_str(), &err);
                CHECK_EXPR_RETURN_FAILURE(
                    CL_SUCCESS == err, "clCreateKernel() fail: %d (%s)", err, kernel_name.c_str());
                auto deleter = [](cl_kernel kernel) { clReleaseKernel(kernel); };
                cl_shared_kernel.reset(opencl_kernel, deleter);
                kernels_[kernel_name] = cl_shared_kernel;
            }
        }
    }
    LOGI(EDEN_CL, "Progress: [%d / %d]", nkernel, nkernel);

    return Status::SUCCESS;
}

CLRuntime::~CLRuntime() { release(); }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
