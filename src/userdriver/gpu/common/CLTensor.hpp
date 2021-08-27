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

#ifndef USERDRIVER_GPU_CL_OPERATORS_CL_TENSOR_HPP_
#define USERDRIVER_GPU_CL_OPERATORS_CL_TENSOR_HPP_

#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"
#include "userdriver/gpu/common/CLBuffer.hpp"
#include "userdriver/gpu/common/CLIncludes.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#define MAXROWPIXEL 60
namespace enn {
namespace ud {
namespace gpu {
#define ENABLERPRINT true
#define ENABLECLTENSORPRINT true

static std::map<std::string, DataType> data_type_map_ = {
    {typeid(float).name(), DataType::FLOAT},
    {typeid(half_float::half).name(), DataType::HALF},
    {typeid(int32_t).name(), DataType::INT32},
    {typeid(uint32_t).name(), DataType::INT32},
    {typeid(int8_t).name(), DataType::INT8},
    {typeid(uint8_t).name(), DataType::UINT8},
    {typeid(bool).name(), DataType::BOOL},
    {typeid(int16_t).name(), DataType::INT16},
    {typeid(uint16_t).name(), DataType::UINT16}};  // default is DataType::FLOAT(0)

// ToDo(all): reduce LOC score
class CLTensor : public ITensor {
public:
    CLTensor(const std::shared_ptr<CLRuntime> runtime,
             const PrecisionType &precision,
             const DataType &data_type,
             const Dim4 &dim,
             const DataOrder &data_order = DataOrder::NCHW,
             const float &scale = 1.0,
             const int32_t &zero_point = 0,
             const BufferType &buffer_type = BufferType::DEDICATED,
             const StorageType &storage_type = StorageType::BUFFER,
             const uint32_t buffer_data_type = 0,
             const int32_t buffer_index = UNDEFINED);

    template <typename T>
    CLTensor(const std::shared_ptr<CLRuntime> runtime,
             const PrecisionType &precision,
             T *data,
             const Dim4 &dims,
             const DataOrder &data_order = DataOrder::NCHW,
             const float &scale = 1.0,
             const int32_t &zero_point = 0,
             const BufferType &buffer_type = BufferType::DEDICATED,
             const StorageType &storage_type = StorageType::BUFFER,
             const uint32_t buffer_data_type = 0,
             const int32_t buffer_index = UNDEFINED) :
        runtime_(runtime),
        precision_(precision), data_type_(data_type_map_[typeid(T).name()]), dims_(dims),
        vdims_({dims.n, dims.c, dims.h, dims.w}), order_(data_order), scale_(scale), zero_point_(zero_point),
        buffer_type_(buffer_type), storage_type_(storage_type), buffer_data_type_(buffer_data_type),
        buffer_index_(buffer_index), is_const_(true) {
        DEBUG_PRINT("CLTensor::CLTensor() is called size is %d", getTotalSizeFromDims());

        if (storage_type_ == StorageType::TEXTURE) {
            TextureDescriptor texture_descriptor;
            texture_descriptor.image_height = dims.h;
            texture_descriptor.image_width = dims.w * dims.n;
            texture_descriptor.bytes = getNumOfBytes();
            texture_descriptor.precision = precision_;
            DataOrderChangeType data_order_change_type =
                order_ == DataOrder::NHWC ? DataOrderChangeType::NHWC2DHWC4 : DataOrderChangeType::NCHW2DHWC4;
            buf_ = std::make_shared<CLBuffer>(texture_descriptor.bytes);
            buf_->assignBuffer(runtime_->allocTexture2D(texture_descriptor));
            writeData(data, false, data_order_change_type);
        } else {
            if (typeid(T) == typeid(float) || typeid(T) == typeid(half_float::half)) {
                buf_ = std::make_shared<CLBuffer>(getNumOfBytes());
                buf_->assignBuffer(runtime_->allocBuffer(getNumOfBytes(), false));
                writeData(data, false);
            } else {
                auto type_bytes = getTypeBytes(data_type_, precision_);
                buf_ = std::make_shared<CLBuffer>(getNumOfBytes());
                buf_->assignBuffer(runtime_->allocBuffer(getNumOfBytes(), false));
                Status state = runtime_->writeBuffer(buf_->getDataPtr(), data, type_bytes, getTotalSizeFromDims(), false);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLRuntime::writeBuffer failed.\n");
            }
        }
    }

    CLTensor(const std::shared_ptr<CLRuntime> runtime,
             const PrecisionType &precision,
             const DataType &data_type,
             const NDims &dims,
             const DataOrder &data_order = DataOrder::NCHW,
             const float &scale = 1.0,
             const int32_t &zero_point = 0,
             const BufferType &buffer_type = BufferType::DEDICATED,
             const StorageType &storage_type = StorageType::BUFFER,
             const uint32_t buffer_data_type = 0,
             const int32_t buffer_index = UNDEFINED);

    template <typename T>
    CLTensor(const std::shared_ptr<CLRuntime> runtime,
             const PrecisionType &precision,
             T *data,
             const NDims &ndim,
             const DataOrder &data_order = DataOrder::NCHW,
             const float &scale = 1.0,
             const int32_t &zero_point = 0,
             const BufferType &buffer_type = BufferType::DEDICATED,
             const StorageType &storage_type = StorageType::BUFFER,
             const uint32_t buffer_data_type = 0,
             const int32_t buffer_index = UNDEFINED) :
        runtime_(runtime),
        precision_(precision), data_type_(data_type_map_[typeid(T).name()]), dims_(extendToDim4(ndim)), vdims_(ndim),
        order_(data_order), scale_(scale), zero_point_(zero_point), buffer_type_(buffer_type), storage_type_(storage_type),
        buffer_data_type_(buffer_data_type), buffer_index_(buffer_index), is_const_(true) {
        DEBUG_PRINT("CLTensor::CLTensor() is called size is %d", getTotalSizeFromDims());

        if (storage_type_ == StorageType::TEXTURE) {
            TextureDescriptor texture_descriptor;
            texture_descriptor.image_height = dims_.h;
            texture_descriptor.image_width = dims_.w * dims_.n;
            texture_descriptor.bytes = getNumOfBytes();
            texture_descriptor.precision = precision_;
            DataOrderChangeType data_order_change_type =
                order_ == DataOrder::NHWC ? DataOrderChangeType::NHWC2DHWC4 : DataOrderChangeType::NCHW2DHWC4;
            buf_ = std::make_shared<CLBuffer>(texture_descriptor.bytes);
            buf_->assignBuffer(runtime_->allocTexture2D(texture_descriptor));
            writeData(data, false, data_order_change_type);
        } else {
            if (typeid(T) == typeid(float) || typeid(T) == typeid(half_float::half)) {
                buf_ = std::make_shared<CLBuffer>(getNumOfBytes());
                buf_->assignBuffer(runtime_->allocBuffer(getNumOfBytes(), false));
                writeData(data, false);
            } else {
                auto type_bytes = getTypeBytes(data_type_, precision_);
                buf_ = std::make_shared<CLBuffer>(getNumOfBytes());
                buf_->assignBuffer(runtime_->allocBuffer(getNumOfBytes(), false));
                Status state = runtime_->writeBuffer(buf_->getDataPtr(), data, type_bytes, getTotalSizeFromDims(), false);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLRuntime::writeBuffer failed.\n");
            }
        }
    }

public:
    size_t getNumOfBytes() override { return getTypeBytes(data_type_, precision_) * (size_t)getTotalSizeFromDims(); }

    DataType getDataType() override { return data_type_; }

    PrecisionType getPrecisionType() override { return precision_; }

    const Dim4 &getDim() override { return dims_; }

    const NDims &getDims() override { return vdims_; }

    uint32_t getDims(const int32_t &idx) override {
        int32_t vdims_size = static_cast<int32_t>(vdims_.size());
        if (idx > vdims_size - 1 || idx + vdims_size < 0) {
            ERROR_PRINT("Invalid idx: %d", idx);
            return 0;
        }
        int32_t m_idx = idx < 0 ? (idx + vdims_size) : idx;
        return vdims_.at(m_idx);
    }

    uint32_t getNumOfDims() override { return vdims_.size(); }

    DataOrder getDataOrder() override { return order_; }

    Status readData(DataPtr data,
                    bool blocking = true,
                    DataOrderChangeType type = DataOrderChangeType::OTHER,
                    void *event = nullptr) override;
    Status writeData(DataPtr data, bool blocking = true, DataOrderChangeType type = DataOrderChangeType::OTHER) override;

    Status reconfigure(const PrecisionType &precision,
                       const DataOrder &order,
                       const Dim4 &dims,
                       const float &scale,
                       const int32_t &offset) override;

    Status reconfigure(const PrecisionType &precision,
                       const DataOrder &order,
                       const NDims &ndim,
                       const float &scale,
                       const int32_t &offset) override;

    Status reorder(const DataOrder &order) override;

    void queryTensorInfo() override;
    void dumpTensorData(std::string file_name = "") override;

    Status reconfigureDim(const Dim4 &dims) override;
    Status reconfigureDims(const NDims &ndim) override;
    Status reconfigureDimAndBuffer(const Dim4 &dims) override;
    Status reconfigureDimsAndBuffer(const NDims &ndim) override;
    Status convertToNHWC(std::shared_ptr<CLTensor> output);
    Status convertToNCHW(std::shared_ptr<CLTensor> output);

    Status resetInterBuffer() override;

    cl_mem getDataPtr() { return buf_->getDataPtr(); }

    Status broadCast(const cl_mem &from_mem, const Dim4 &to_dim);
    Status broadCastTo(std::shared_ptr<CLTensor> output);
    Status broadCastTexture2d(const cl_mem &in_mem, Dim4 dim);
    Status nchwBlockBroadCastTo(std::shared_ptr<CLTensor> output);
    Status convertInt2Float();
    Status convertFloat2Int();
    uint32_t getTotalSizeFromDims() {
        return vdims_.size() ? std::accumulate(vdims_.begin(), vdims_.end(), 1u, std::multiplies<uint32_t>()) : 0;
    }

    float getScale() { return scale_; }

    void setScale(float scale) { scale_ = scale; }

    int32_t getZeroPoint() { return zero_point_; }

    void setZeroPoint(int32_t zero_point) { zero_point_ = zero_point; }

    int32_t get_buffer_index() override { return buffer_index_; }

    void set_buffer_index(int32_t index) override { buffer_index_ = index; }

    uint32_t get_buffer_type() override { return buffer_data_type_; }

    void set_buffer_type(uint32_t type) override { buffer_data_type_ = type; }

    void set_buffer_ptr(void *addr) override { raw_buffer = addr; }

    bool is_const() override { return is_const_; }

    void setOffset(const int32_t &offset) { offset_ = offset; }

    void print(std::string name);
    static void checknull(std::shared_ptr<CLTensor> cltensor, std::string name) {
        if (cltensor == nullptr)
            ITensor::placeholder(name + " IS nullptr", ColorType::RED);
        else {
            ITensor::placeholder(name + " NOT nullptr", ColorType::GREEN);
            cltensor->print(name);
        }
        return;
    }
    int32_t getOffset() { return offset_; }

    uint32_t getDepth() { return 1; }

    uint32_t getSlice() {
        int div = 4;
        return IntegralDivideRoundUp(dims_.c, div);
    }

    uint32_t getImageH() { return dims_.h * getSlice(); }

    uint32_t getImageW() { return dims_.w * dims_.n; }

    ~CLTensor() {
        vdims_.clear();
        if (buf_->getDataPtr() != nullptr) {
            Status ret = runtime_->releaseBuffer(buf_);
            if (ret != Status::SUCCESS) {
                ERROR_PRINT("~CLTensor(): buffer release fail\n");
            }
        }
    }

private:
    template <typename T> DataType getDataTypeFromTemplate();
    template <typename T> void dumpDataToStream(std::ofstream &ofs, T *data, const uint32_t &num) {
        int coutw = 0, numsperrow = 0;
        if (ENABLERPRINT) {
            for (uint32_t i = 0; i < num; i++) {
                if (typeid(T) == typeid(float) || typeid(T) == typeid(half_float::half))
                    coutw =
                        coutw < std::to_string((float)data[i]).length() ? std::to_string((float)data[i]).length() : coutw;
                else {
                    coutw = coutw < std::to_string((int)data[i]).length() ? std::to_string((int)data[i]).length() : coutw;
                }
            }
            numsperrow = MAXROWPIXEL / (coutw + 1);
        }
        for (uint32_t i = 0; i < num; i++) {
            if (typeid(T) == typeid(float) || typeid(T) == typeid(half_float::half))
                ofs << (float)data[i] << std::endl;
            else {
                ofs << (int)data[i] << std::endl;
            }
            if (ENABLERPRINT) {
                if (typeid(T) == typeid(float) || typeid(T) == typeid(half_float::half))
                    std::cout << std::setw(coutw + 1) << std::setiosflags(std::ios::right) << (float)data[i];
                else {
                    std::cout << std::setw(coutw + 1) << std::setiosflags(std::ios::right) << (int)data[i];
                }
                if ((i + 1) % numsperrow == 0)
                    std::cout << std::endl;
            }
        }
    }

private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    DataType data_type_;
    Dim4 dims_;
    NDims vdims_;
    DataOrder order_;

    float scale_ = 0.0f;      // quantization param
    int32_t zero_point_ = 0;  // quantization param

    BufferType buffer_type_;
    StorageType storage_type_;

    uint32_t buffer_data_type_ = 0;
    int32_t buffer_index_ = UNDEFINED;
    bool is_const_ = false;

    std::shared_ptr<CLBuffer> buf_;
    int32_t offset_ = 0;  // memory offset for gpu buffer
};                        // class CLTensor

}  // namespace gpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_GPU_CL_OPERATORS_CL_TENSOR_HPP_
