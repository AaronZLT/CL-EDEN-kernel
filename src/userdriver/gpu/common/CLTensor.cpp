#include "CLTensor.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLTensor::CLTensor(const std::shared_ptr<CLRuntime> runtime,
                   const PrecisionType &precision,
                   const DataType &data_type,
                   const Dim4 &dims,
                   const DataOrder &data_order,
                   const float &scale,
                   const int32_t &zero_point,
                   const BufferType &buffer_type,
                   const StorageType &storage_type,
                   const uint32_t buffer_data_type,
                   const int32_t buffer_index) :
    runtime_(runtime),
    precision_(precision), data_type_(data_type), dims_(dims), vdims_({dims.n, dims.c, dims.h, dims.w}), order_(data_order),
    scale_(scale), zero_point_(zero_point), buffer_type_(buffer_type), storage_type_(storage_type),
    buffer_data_type_(buffer_data_type), buffer_index_(buffer_index), is_const_(false) {
    DEBUG_PRINT("CLTensor::CLTensor() is called size is %d", getTotalSizeFromDims());

    buf_ = runtime_->getBuffer(precision_, data_type_, dims_, buffer_type, storage_type);
}

CLTensor::CLTensor(const std::shared_ptr<CLRuntime> runtime,
                   const PrecisionType &precision,
                   const DataType &data_type,
                   const NDims &ndim,
                   const DataOrder &data_order,
                   const float &scale,
                   const int32_t &zero_point,
                   const BufferType &buffer_type,
                   const StorageType &storage_type,
                   const uint32_t buffer_data_type,
                   const int32_t buffer_index) :
    runtime_(runtime),
    precision_(precision), data_type_(data_type), dims_(extendToDim4(ndim)), vdims_(ndim), order_(data_order), scale_(scale),
    zero_point_(zero_point), buffer_type_(buffer_type), storage_type_(storage_type), buffer_data_type_(buffer_data_type),
    buffer_index_(buffer_index), is_const_(false) {
    DEBUG_PRINT("CLTensor::CLTensor() is called size is %d", getTotalSizeFromDims());

    buf_ = runtime_->getBuffer(precision_, data_type_, dims_, buffer_type, storage_type);
}

Status CLTensor::resetInterBuffer() { return runtime_->resetInterBuffer(buf_); }

Status CLTensor::reconfigure(const PrecisionType &precision,
                             const DataOrder &order,
                             const Dim4 &dims,
                             const float &scale,
                             const int32_t &offset) {
    DEBUG_PRINT("CLTensor::reconfigure() is called");
    DEBUG_PRINT(
        "dim change %u %u %u %u --> %u %u %u %u\n", dims_.n, dims_.c, dims_.h, dims_.w, dims.n, dims.c, dims.h, dims.w);
    precision_ = precision;
    order_ = order;
    dims_ = dims;
    vdims_ = {dims.n, dims.c, dims.h, dims.w};
    scale_ = scale;
    zero_point_ = offset;
    runtime_->releaseBuffer(buf_);
    buf_->assignBuffer(runtime_->allocBuffer(getNumOfBytes(), true));
    return Status::SUCCESS;
}

Status CLTensor::reconfigure(const PrecisionType &precision,
                             const DataOrder &order,
                             const NDims &ndim,
                             const float &scale,
                             const int32_t &offset) {
    DEBUG_PRINT("CLTensor::reconfigure() is called");
    DEBUG_PRINT("ndim change %u %u %u %u --> %u %u %u %u\n",
                viewNDimsAt(vdims_, 0),
                viewNDimsAt(vdims_, 1),
                viewNDimsAt(vdims_, 2),
                viewNDimsAt(vdims_, 3),
                viewNDimsAt(ndim, 0),
                viewNDimsAt(ndim, 1),
                viewNDimsAt(ndim, 2),
                viewNDimsAt(ndim, 3));
    precision_ = precision;
    order_ = order;
    vdims_ = ndim;
    dims_ = extendToDim4(vdims_);
    scale_ = scale;
    zero_point_ = offset;
    runtime_->releaseBuffer(buf_);
    buf_->assignBuffer(runtime_->allocBuffer(getNumOfBytes(), true));
    return Status::SUCCESS;
}

Status CLTensor::reorder(const DataOrder &order) {
    DEBUG_PRINT("CLTensor::reorder() is not supported yet");
    order_ = order;
    return Status::FAILURE;
}

Status CLTensor::reconfigureDim(const Dim4 &dims) {
    DEBUG_PRINT("CLTensor::reconfigureDim() is called");
    DEBUG_PRINT(
        "dim change %u %u %u %u --> %u %u %u %u\n", dims_.n, dims_.c, dims_.h, dims_.w, dims.n, dims.c, dims.h, dims.w);
    dims_ = dims;
    vdims_ = {dims.n, dims.c, dims.h, dims.w};
    return Status::SUCCESS;
}

Status CLTensor::reconfigureDims(const NDims &ndim) {
    DEBUG_PRINT("CLTensor::reconfigureDims() is called");
    DEBUG_PRINT("ndim change %u %u %u %u --> %u %u %u %u\n",
                viewNDimsAt(vdims_, 0),
                viewNDimsAt(vdims_, 1),
                viewNDimsAt(vdims_, 2),
                viewNDimsAt(vdims_, 3),
                viewNDimsAt(ndim, 0),
                viewNDimsAt(ndim, 1),
                viewNDimsAt(ndim, 2),
                viewNDimsAt(ndim, 3));
    vdims_ = ndim;
    dims_ = extendToDim4(vdims_);
    return Status::SUCCESS;
}

Status CLTensor::reconfigureDimAndBuffer(const Dim4 &dims) {
    DEBUG_PRINT("CLTensor::reconfigureDimAndBuffer() is called");
    DEBUG_PRINT(
        "dim change %u %u %u %u --> %u %u %u %u\n", dims_.n, dims_.c, dims_.h, dims_.w, dims.n, dims.c, dims.h, dims.w);
    uint32_t size_ori = getTotalSizeFromDims();
    dims_ = dims;
    vdims_ = {dims.n, dims.c, dims.h, dims.w};
    uint32_t size_cur = getTotalSizeFromDims();
    auto bytes = getNumOfBytes();
    if (size_ori != size_cur) {
        if (size_ori != 0) {
            runtime_->releaseBuffer(buf_);
        }
        if (size_cur != 0) {
            buf_ = std::make_shared<CLBuffer>(bytes);
            buf_->assignBuffer(runtime_->allocBuffer(bytes, true));
        }
    }
    return Status::SUCCESS;
}

Status CLTensor::reconfigureDimsAndBuffer(const NDims &ndim) {
    DEBUG_PRINT("CLTensor::reconfigureDimsAndBuffer() is called");
    DEBUG_PRINT("ndim change %u %u %u %u --> %u %u %u %u\n",
                viewNDimsAt(vdims_, 0),
                viewNDimsAt(vdims_, 1),
                viewNDimsAt(vdims_, 2),
                viewNDimsAt(vdims_, 3),
                viewNDimsAt(ndim, 0),
                viewNDimsAt(ndim, 1),
                viewNDimsAt(ndim, 2),
                viewNDimsAt(ndim, 3));
    uint32_t size_ori = getTotalSizeFromDims();
    vdims_ = ndim;
    dims_ = extendToDim4(vdims_);
    uint32_t size_cur = getTotalSizeFromDims();
    auto bytes = getNumOfBytes();
    if (size_ori != size_cur) {
        if (size_ori != 0) {
            runtime_->releaseBuffer(buf_);
        }
        if (size_cur != 0) {
            buf_ = std::make_shared<CLBuffer>(bytes);
            buf_->assignBuffer(runtime_->allocBuffer(bytes, true));
        }
    }
    return Status::SUCCESS;
}

/**
 * @brief Convert Layout of EDEN Buffer from NCHW to NHWC
 * @details This function converts layout of CLTensor (Buffer
 * and Dimensions) from NCHW to NHWC.
 */
Status CLTensor::convertToNHWC(std::shared_ptr<CLTensor> output) {
    DEBUG_PRINT("CLTensor::convertNchwToNhwc() is called");
    Dim4 nchw = getDim();
    Dim4 nhwc = {nchw.n, nchw.h, nchw.w, nchw.c};
    if (!isDimsSame(nhwc, output->getDim())) {
        DEBUG_PRINT("nhwc dim != output dim");
        output->reconfigureDim(nhwc);
    }
    if (getTotalSizeFromDims() == 0) {
        return Status::SUCCESS;
    }
    DataType mdata_type = data_type_;
    if (data_type_ == DataType::FLOAT && precision_ == PrecisionType::FP16) {
        mdata_type = DataType::HALF;
    } else if (data_type_ == DataType::HALF && precision_ == PrecisionType::FP32) {
        mdata_type = DataType::FLOAT;
    }
    return runtime_->NCHW2NHWC(buf_->getDataPtr(), output->getDataPtr(), nchw, mdata_type, PrecisionChangeMode::OTHER);
}

/**
 * @brief Convert Layout of EDEN Buffer from NHWC to NCHW
 * @details This function converts layout of CLTensor (Buffer
 * and Dimensions) from NCHW to NHWC.
 */
Status CLTensor::convertToNCHW(std::shared_ptr<CLTensor> output) {
    DEBUG_PRINT("CLTensor::convertNhwcToNchw() is called");
    Dim4 nhwc = getDim();
    Dim4 nchw = {nhwc.n, nhwc.w, nhwc.c, nhwc.h};
    if (!isDimsSame(nchw, output->getDim())) {
        DEBUG_PRINT("nchw dim != output dim");
        output->reconfigureDim(nchw);
    }
    if (getTotalSizeFromDims() == 0) {
        return Status::SUCCESS;
    }
    DataType mdata_type = data_type_;
    if (data_type_ == DataType::FLOAT && precision_ == PrecisionType::FP16) {
        mdata_type = DataType::HALF;
    } else if (data_type_ == DataType::HALF && precision_ == PrecisionType::FP32) {
        mdata_type = DataType::FLOAT;
    }

    return runtime_->NHWC2NCHW(buf_->getDataPtr(), output->getDataPtr(), nchw, mdata_type, PrecisionChangeMode::OTHER);
}

Status CLTensor::writeData(DataPtr data, bool blocking, DataOrderChangeType type) {
    DEBUG_PRINT("CLTensor::writeData() is called.");
    CHECK_EXPR_RETURN_FAILURE(buf_->getDataPtr() != nullptr, "CLTensor::writeData() fail");
    CHECK_EXPR_RETURN_FAILURE(data != nullptr, "data is nullptr");
    auto type_bytes = getTypeBytes(data_type_, precision_);
    auto num = getTotalSizeFromDims();
    Status state = Status::SUCCESS;
    if (data_type_ == DataType::FLOAT && precision_ == PrecisionType::FP16) {
        auto fp32_bytes = 2 * type_bytes * (size_t)num;
        std::shared_ptr<CLBuffer> middle_fp32_buffer = std::make_shared<CLBuffer>(fp32_bytes);
        middle_fp32_buffer->assignBuffer(runtime_->allocBuffer(fp32_bytes, false, data));
        if (type == DataOrderChangeType::NHWC2NCHW) {
            state = runtime_->NHWC2NCHW(middle_fp32_buffer->getDataPtr(),
                                        buf_->getDataPtr(),
                                        getDim(),
                                        data_type_,
                                        PrecisionChangeMode::FP32_TO_FP16);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() NHWC2NCHW failed.\n");
        } else if (type == DataOrderChangeType::NCHW2NHWC) {
            state = runtime_->NCHW2NHWC(middle_fp32_buffer->getDataPtr(),
                                        buf_->getDataPtr(),
                                        getDim(),
                                        data_type_,
                                        PrecisionChangeMode::FP32_TO_FP16);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() NCHW2NHWC failed.\n");
        } else if (type == DataOrderChangeType::NHWC2DHWC4 && storage_type_ == StorageType::TEXTURE) {
            state = runtime_->NHWC2DHWC4(middle_fp32_buffer->getDataPtr(),
                                         buf_->getDataPtr(),
                                         getDim(),
                                         data_type_,
                                         PrecisionChangeMode::FP32_TO_FP16);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() runtime_->NHWC2DHWC4 failed.\n");
        } else if (type == DataOrderChangeType::NCHW2DHWC4 && storage_type_ == StorageType::TEXTURE) {
            state = runtime_->NCHW2DHWC4(middle_fp32_buffer->getDataPtr(),
                                         buf_->getDataPtr(),
                                         getDim(),
                                         data_type_,
                                         PrecisionChangeMode::FP32_TO_FP16);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() runtime_->NCHW2DHWC4 failed.\n");
        } else {
            if (storage_type_ == StorageType::TEXTURE) {
                DEBUG_PRINT("writeData: write into texture2d without dataorder change is not supported yet");
                return Status::FAILURE;
            } else {
                state = runtime_->copyFloat2Half(buf_->getDataPtr(), middle_fp32_buffer->getDataPtr(), num);
            }
        }
        runtime_->releaseBuffer(middle_fp32_buffer);
    } else if (data_type_ == DataType::HALF && precision_ == PrecisionType::FP32) {
        auto fp16_bytes = type_bytes * (size_t)num;
        std::shared_ptr<CLBuffer> middle_fp16_buffer = std::make_shared<CLBuffer>(fp16_bytes);
        middle_fp16_buffer->assignBuffer(runtime_->allocBuffer(fp16_bytes, false));
        state = runtime_->writeBuffer(middle_fp16_buffer->getDataPtr(), data, type_bytes / 2, num, blocking);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() writeBuffer failed.\n");
        if (type == DataOrderChangeType::NHWC2NCHW) {
            runtime_->NHWC2NCHW(middle_fp16_buffer->getDataPtr(),
                                buf_->getDataPtr(),
                                getDim(),
                                data_type_,
                                PrecisionChangeMode::FP16_TO_FP32);
        } else if (type == DataOrderChangeType::NCHW2NHWC) {
            state = runtime_->NCHW2NHWC(middle_fp16_buffer->getDataPtr(),
                                        buf_->getDataPtr(),
                                        getDim(),
                                        data_type_,
                                        PrecisionChangeMode::FP16_TO_FP32);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() NCHW2NHWC failed.\n");
        } else if (type == DataOrderChangeType::NHWC2DHWC4 && storage_type_ == StorageType::TEXTURE) {
            state = runtime_->NHWC2DHWC4(middle_fp16_buffer->getDataPtr(),
                                         buf_->getDataPtr(),
                                         getDim(),
                                         data_type_,
                                         PrecisionChangeMode::FP16_TO_FP32);
        } else if (type == DataOrderChangeType::NCHW2DHWC4 && storage_type_ == StorageType::TEXTURE) {
            state = runtime_->NCHW2DHWC4(middle_fp16_buffer->getDataPtr(),
                                         buf_->getDataPtr(),
                                         getDim(),
                                         data_type_,
                                         PrecisionChangeMode::FP16_TO_FP32);
        } else {
            if (storage_type_ == StorageType::TEXTURE) {
                DEBUG_PRINT("writeData: write into texture2d without dataorder change is not supported yet");
                return Status::FAILURE;
            } else {
                state = runtime_->copyHalf2Float(buf_->getDataPtr(), middle_fp16_buffer->getDataPtr(), num);
            }
        }
        runtime_->releaseBuffer(middle_fp16_buffer);
    } else {
        auto bytes = type_bytes * (size_t)num;
        if (type == DataOrderChangeType::NHWC2NCHW) {
            std::shared_ptr<CLBuffer> middle_buffer = std::make_shared<CLBuffer>(bytes);
            middle_buffer->assignBuffer(runtime_->allocBuffer(bytes, false));
            state = runtime_->writeBuffer(middle_buffer->getDataPtr(), data, type_bytes, num, blocking);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() writeBuffer failed.\n");
            state = runtime_->NHWC2NCHW(
                middle_buffer->getDataPtr(), buf_->getDataPtr(), getDim(), data_type_, PrecisionChangeMode::OTHER);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() NHWC2NCHW failed.\n");
            runtime_->releaseBuffer(middle_buffer);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::releaseBuffer() failed.\n");
        } else if (type == DataOrderChangeType::NCHW2NHWC) {
            std::shared_ptr<CLBuffer> middle_buffer = std::make_shared<CLBuffer>(bytes);
            middle_buffer->assignBuffer(runtime_->allocBuffer(bytes, false));
            state = runtime_->writeBuffer(middle_buffer->getDataPtr(), data, type_bytes, num, blocking);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() writeBuffer failed.\n");
            state = runtime_->NCHW2NHWC(
                middle_buffer->getDataPtr(), buf_->getDataPtr(), getDim(), data_type_, PrecisionChangeMode::OTHER);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() NCHW2NHWC failed.\n");
            state = runtime_->releaseBuffer(middle_buffer);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() releaseBuffer failed.\n");
        } else if (type == DataOrderChangeType::NHWC2DHWC4 && storage_type_ == StorageType::TEXTURE) {
            std::shared_ptr<CLBuffer> middle_buffer = std::make_shared<CLBuffer>(bytes);
            middle_buffer->assignBuffer(runtime_->allocBuffer(bytes, false));
            state = runtime_->writeBuffer(middle_buffer->getDataPtr(), data, type_bytes, num, blocking);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() writeBuffer failed.\n");
            state = runtime_->NHWC2DHWC4(
                middle_buffer->getDataPtr(), buf_->getDataPtr(), getDim(), data_type_, PrecisionChangeMode::OTHER);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() NHWC2DHWC4 failed.\n");
            state = runtime_->releaseBuffer(middle_buffer);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() releaseBuffer failed.\n");
        } else if (type == DataOrderChangeType::NCHW2DHWC4 && storage_type_ == StorageType::TEXTURE) {
            std::shared_ptr<CLBuffer> middle_buffer = std::make_shared<CLBuffer>(bytes);
            middle_buffer->assignBuffer(runtime_->allocBuffer(bytes, false));
            state = runtime_->writeBuffer(middle_buffer->getDataPtr(), data, type_bytes, num, blocking);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() writeBuffer failed.\n");
            state = runtime_->NCHW2DHWC4(
                middle_buffer->getDataPtr(), buf_->getDataPtr(), getDim(), data_type_, PrecisionChangeMode::OTHER);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() NCHW2DHWC4 failed.\n");
            state = runtime_->releaseBuffer(middle_buffer);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::writeData() releaseBuffer failed.\n");
        } else {
            if (storage_type_ == StorageType::TEXTURE) {
                DEBUG_PRINT("writeData: write into texture2d without dataorder change is not supported yet");
                return Status::FAILURE;
            } else {
                state = runtime_->writeBuffer(buf_->getDataPtr(), data, type_bytes, getTotalSizeFromDims(), blocking);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::writeData() writeBuffer failed.\n");
            }
        }
    }
    return state;
}

Status CLTensor::readData(DataPtr result, bool blocking, DataOrderChangeType type, void *event) {
    DEBUG_PRINT("CLTensor::readData() is called");
    CHECK_EXPR_RETURN_FAILURE(buf_->getDataPtr() != nullptr, "CLTensor::readData() fail");
    auto type_bytes = getTypeBytes(data_type_, precision_);
    auto num = getTotalSizeFromDims();
    Status state = Status::SUCCESS;
    if (data_type_ == DataType::FLOAT && precision_ == PrecisionType::FP16) {
        auto fp32_bytes = 2 * type_bytes * (size_t)num;
        std::shared_ptr<CLBuffer> middle_fp32_buffer = std::make_shared<CLBuffer>(fp32_bytes);
        middle_fp32_buffer->assignBuffer(runtime_->allocBuffer(fp32_bytes, false));
        if (type == DataOrderChangeType::NCHW2NHWC) {
            state = runtime_->NCHW2NHWC(buf_->getDataPtr(),
                                        middle_fp32_buffer->getDataPtr(),
                                        getDim(),
                                        data_type_,
                                        PrecisionChangeMode::FP16_TO_FP32);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() NCHW2NHWC failed.\n");
        } else if (type == DataOrderChangeType::NHWC2NCHW) {
            state = runtime_->NHWC2NCHW(buf_->getDataPtr(),
                                        middle_fp32_buffer->getDataPtr(),
                                        getDim(),
                                        data_type_,
                                        PrecisionChangeMode::FP16_TO_FP32);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() NHWC2NCHW failed.\n");
        } else if (type == DataOrderChangeType::DHWC42NCHW && storage_type_ == StorageType::TEXTURE) {
            state = runtime_->DHWC42NCHW(buf_->getDataPtr(),
                                         middle_fp32_buffer->getDataPtr(),
                                         getDim(),
                                         data_type_,
                                         PrecisionChangeMode::FP16_TO_FP32);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() DHWC42NCHW failed.\n");
        } else if (type == DataOrderChangeType::DHWC42NHWC && storage_type_ == StorageType::TEXTURE) {
            state = runtime_->DHWC42NHWC(buf_->getDataPtr(),
                                         middle_fp32_buffer->getDataPtr(),
                                         getDim(),
                                         data_type_,
                                         PrecisionChangeMode::FP16_TO_FP32);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() DHWC42NHWC failed.\n");
        } else {
            if (storage_type_ == StorageType::TEXTURE) {
                runtime_->copyHalf2FloatTexture2D(middle_fp32_buffer->getDataPtr(), buf_->getDataPtr(), dims_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() copyHalf2FloatTexture2D failed.\n");
            } else {
                state = runtime_->copyHalf2Float(middle_fp32_buffer->getDataPtr(), buf_->getDataPtr(), num);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() copyHalf2Float failed.\n");
            }
        }
        state = runtime_->readBuffer(result, middle_fp32_buffer->getDataPtr(), 2 * type_bytes, num, blocking, event);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() readBuffer failed.\n");
        state = runtime_->releaseBuffer(middle_fp32_buffer);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() releaseBuffer failed.\n");
    } else if (data_type_ == DataType::HALF && precision_ == PrecisionType::FP32) {
        auto fp16_bytes = type_bytes * (size_t)num;
        std::shared_ptr<CLBuffer> middle_fp16_buffer = std::make_shared<CLBuffer>(fp16_bytes);
        middle_fp16_buffer->assignBuffer(runtime_->allocBuffer(fp16_bytes, false));
        if (type == DataOrderChangeType::NCHW2NHWC) {
            state = runtime_->NCHW2NHWC(buf_->getDataPtr(),
                                        middle_fp16_buffer->getDataPtr(),
                                        getDim(),
                                        data_type_,
                                        PrecisionChangeMode::FP32_TO_FP16);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() NCHW2NHWC failed.\n");
        } else if (type == DataOrderChangeType::NHWC2NCHW) {
            state = runtime_->NHWC2NCHW(buf_->getDataPtr(),
                                        middle_fp16_buffer->getDataPtr(),
                                        getDim(),
                                        data_type_,
                                        PrecisionChangeMode::FP32_TO_FP16);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() NHWC2NCHW failed.\n");
        } else if (type == DataOrderChangeType::DHWC42NCHW && storage_type_ == StorageType::TEXTURE) {
            state = runtime_->DHWC42NCHW(buf_->getDataPtr(),
                                         middle_fp16_buffer->getDataPtr(),
                                         getDim(),
                                         data_type_,
                                         PrecisionChangeMode::FP32_TO_FP16);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() DHWC42NCHW failed.\n");
        } else if (type == DataOrderChangeType::DHWC42NHWC && storage_type_ == StorageType::TEXTURE) {
            state = runtime_->DHWC42NHWC(buf_->getDataPtr(),
                                         middle_fp16_buffer->getDataPtr(),
                                         getDim(),
                                         data_type_,
                                         PrecisionChangeMode::FP32_TO_FP16);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() DHWC42NHWC failed.\n");
        } else {
            if (storage_type_ == StorageType::TEXTURE) {
                state = runtime_->copyFloat2HalfTexture2D(middle_fp16_buffer->getDataPtr(), buf_->getDataPtr(), dims_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() copyFloat2HalfTexture2D failed.\n");
            } else {
                state = runtime_->copyFloat2Half(middle_fp16_buffer->getDataPtr(), buf_->getDataPtr(), num);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() copyFloat2Half failed.\n");
            }
        }
        runtime_->readBuffer(result, middle_fp16_buffer->getDataPtr(), type_bytes / 2, num, blocking, event);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() readBuffer failed.\n");
        runtime_->releaseBuffer(middle_fp16_buffer);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() releaseBuffer failed.\n");
    } else {
        auto bytes = type_bytes * (size_t)num;
        if (type == DataOrderChangeType::NCHW2NHWC) {
            std::shared_ptr<CLBuffer> middle_buffer = std::make_shared<CLBuffer>(bytes);
            middle_buffer->assignBuffer(runtime_->allocBuffer(bytes, false));
            state = runtime_->NCHW2NHWC(
                buf_->getDataPtr(), middle_buffer->getDataPtr(), getDim(), data_type_, PrecisionChangeMode::OTHER);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::readData() NCHW2NHWC failed.\n");
            state = runtime_->readBuffer(result, middle_buffer->getDataPtr(), type_bytes, num, blocking, event);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::readData() readBuffer failed.\n");
            state = runtime_->releaseBuffer(middle_buffer);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::readData() releaseBuffer failed.\n");
        } else if (type == DataOrderChangeType::NHWC2NCHW) {
            std::shared_ptr<CLBuffer> middle_buffer = std::make_shared<CLBuffer>(bytes);
            middle_buffer->assignBuffer(runtime_->allocBuffer(bytes, false));
            state = runtime_->NHWC2NCHW(
                buf_->getDataPtr(), middle_buffer->getDataPtr(), getDim(), data_type_, PrecisionChangeMode::OTHER);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::readData() NHWC2NCHW failed.\n");
            state = runtime_->readBuffer(result, middle_buffer->getDataPtr(), type_bytes, num, blocking, event);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::readData() readBuffer failed.\n");
            state = runtime_->releaseBuffer(middle_buffer);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::readData() releaseBuffer failed.\n");
        } else if (type == DataOrderChangeType::DHWC42NCHW && storage_type_ == StorageType::TEXTURE) {
            std::shared_ptr<CLBuffer> middle_buffer = std::make_shared<CLBuffer>(bytes);
            middle_buffer->assignBuffer(runtime_->allocBuffer(bytes, false));
            state = runtime_->DHWC42NCHW(
                buf_->getDataPtr(), middle_buffer->getDataPtr(), getDim(), data_type_, PrecisionChangeMode::OTHER);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::readData() DHWC42NCHW failed.\n");
            state = runtime_->readBuffer(result, middle_buffer->getDataPtr(), type_bytes, num, blocking, event);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::readData() readBuffer failed.\n");
            state = runtime_->releaseBuffer(middle_buffer);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::readData() releaseBuffer failed.\n");
        } else if (type == DataOrderChangeType::DHWC42NHWC && storage_type_ == StorageType::TEXTURE) {
            std::shared_ptr<CLBuffer> middle_buffer = std::make_shared<CLBuffer>(bytes);
            middle_buffer->assignBuffer(runtime_->allocBuffer(bytes, false));
            state = runtime_->DHWC42NHWC(
                buf_->getDataPtr(), middle_buffer->getDataPtr(), getDim(), data_type_, PrecisionChangeMode::OTHER);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() DHWC42NHWC failed.\n");
            state = runtime_->readBuffer(result, middle_buffer->getDataPtr(), type_bytes, num, blocking, event);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "CLTensor::readData() readBuffer failed.\n");
            state = runtime_->releaseBuffer(middle_buffer);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::readData() releaseBuffer failed.\n");
        } else {
            if (storage_type_ == StorageType::TEXTURE) {
                state = runtime_->readBufferTexture2D(result, buf_->getDataPtr(), dims_, blocking);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::readData() readBufferTexture2D failed.\n");
            } else {
                state = runtime_->readBuffer(result, buf_->getDataPtr(), type_bytes, num, blocking, event);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLTensor::readData() readBuffer failed.\n");
            }
        }
    }
    return state;
}

void CLTensor::queryTensorInfo() {
    std::cout << "-------- Tensor Info -------" << std::endl;
    std::cout << "Precision: " << static_cast<std::underlying_type<PrecisionType>::type>(precision_) << std::endl;
    std::cout << "Order: " << static_cast<std::underlying_type<DataOrder>::type>(order_) << std::endl;
    std::cout << "Shape: ";
    std::cout << "(n: " << dims_.n << ") (c: " << dims_.c << ") (h: " << dims_.h << ") (w: " << dims_.w << ")" << std::endl;
    std::cout << "ZeroPoint: " << zero_point_ << std::endl;
    std::cout << "Scale: " << scale_ << std::endl;
    std::cout << "Buffer: " << buf_->getDataPtr() << std::endl;
    std::cout << "----------------------------" << std::endl;
}

void CLTensor::dumpTensorData(std::string file_name) {
    ENN_DBG_PRINT("dumpTensorData %s\n", file_name.c_str());
    std::ofstream out_file(file_name);
    // if file_name is empty will redirect to std::cout
    if (file_name.empty())
        out_file.basic_ios<char>::rdbuf(std::cout.rdbuf());
    auto total_bytes = (size_t)(getTotalSizeFromDims()) * getTypeBytes(data_type_, precision_);

    if (precision_ == PrecisionType::FP16 && data_type_ == DataType::FLOAT)
        total_bytes *= 2;
    //    else if (precision_ == PrecisionType::FP32 && data_type_ == DataType::HALF) total_bytes /=
    //    2;
    std::shared_ptr<uint8_t> buffer;
    buffer.reset(new uint8_t[total_bytes]);
    readData(buffer.get());

    switch (data_type_) {
    case DataType::FLOAT:
        dumpDataToStream(out_file, reinterpret_cast<float *>(buffer.get()), total_bytes / sizeof(float));
        break;
    case DataType::HALF:
        dumpDataToStream(
            out_file, reinterpret_cast<half_float::half *>(buffer.get()), total_bytes / sizeof(half_float::half));
        break;
    case DataType::INT8:
        dumpDataToStream(out_file, reinterpret_cast<int8_t *>(buffer.get()), total_bytes / sizeof(int8_t));
        break;
    case DataType::INT32:
        dumpDataToStream(out_file, reinterpret_cast<int32_t *>(buffer.get()), total_bytes / sizeof(int32_t));
        break;
    case DataType::UINT8:
        dumpDataToStream(out_file, reinterpret_cast<uint8_t *>(buffer.get()), total_bytes / sizeof(uint8_t));
        break;
    case DataType::INT16:
        dumpDataToStream(out_file, reinterpret_cast<int16_t *>(buffer.get()), total_bytes / sizeof(int16_t));
        break;
    case DataType::UINT16:
        dumpDataToStream(out_file, reinterpret_cast<uint16_t *>(buffer.get()), total_bytes / sizeof(uint16_t));
        break;
    case DataType::BOOL:
        dumpDataToStream(out_file, reinterpret_cast<bool *>(buffer.get()), total_bytes / sizeof(bool));
        break;
    }
    out_file.close();
}

Status CLTensor::broadCast(const cl_mem &from_mem, const Dim4 &to_dim) {
    DEBUG_PRINT("CLTensor::broadCast() is called");
    PrecisionType prec = precision_;
    if (getDataType() == DataType::BOOL || getDataType() == DataType::INT8 || getDataType() == DataType::UINT8) {
        prec = PrecisionType::UINT8;
    } else if (getDataType() == DataType::INT32) {
        prec = PrecisionType::FP32;
    }
    cl_mem to_mem = buf_->getDataPtr();
    const Dim4 &from_dim = getDim();
    size_t global[3] = {to_dim.n, to_dim.c, to_dim.h * to_dim.w};
    size_t local[3] = {1, 1, static_cast<size_t>(findMaxFactor(global[2], 128))};

    std::shared_ptr<struct _cl_kernel> broadcast_kernel = nullptr;
    Status state = runtime_->setKernel(&broadcast_kernel, "broadcast", prec);
    CHECK_EXPR_RETURN_FAILURE(state == Status::SUCCESS, "CLTensor::broadCast() setKernel fail");
    state = runtime_->setKernelArg(
        broadcast_kernel.get(), from_mem, to_mem, to_dim.w, from_dim.n, from_dim.c, from_dim.h, from_dim.w);
    CHECK_EXPR_RETURN_FAILURE(state == Status::SUCCESS, "CLTensor::broadCast() setKernelArg fail");
    state = runtime_->enqueueKernel(broadcast_kernel.get(), 3, global, local);
    CHECK_EXPR_RETURN_FAILURE(state == Status::SUCCESS, "CLTensor::broadCast() enqueueKernel fail");

    return Status::SUCCESS;
}

Status CLTensor::broadCastTo(std::shared_ptr<CLTensor> output) {
    DEBUG_PRINT("CLTensor::broadCastTo() is called");
    PrecisionType precision = precision_;
    if (getDataType() == DataType::BOOL || getDataType() == DataType::INT8 || getDataType() == DataType::UINT8) {
        precision = PrecisionType::UINT8;
    } else if (getDataType() == DataType::INT32) {
        precision = PrecisionType::FP32;
    }
    if (output->getNumOfDims() <= 4) {
        CHECK_EXPR_RETURN_FAILURE(getNumOfDims() <= 4, "Invalid input dims, which are not broadcastable.");
        const cl_mem from_mem = buf_->getDataPtr();
        cl_mem to_mem = output->getDataPtr();
        NDims expanded_from_dims(4 - getNumOfDims(), 1);
        expanded_from_dims.insert(expanded_from_dims.end(), vdims_.begin(), vdims_.end());
        NDims expanded_to_dims(4 - output->getNumOfDims(), 1);
        expanded_to_dims.insert(expanded_to_dims.end(), output->getDims().begin(), output->getDims().end());
        size_t global[3] = {expanded_to_dims[0], expanded_to_dims[1], expanded_to_dims[2] * expanded_to_dims[3]};
        size_t local[3] = {1, 1, static_cast<size_t>(findMaxFactor(global[2], 128))};
        std::shared_ptr<struct _cl_kernel> broadcast_kernel = nullptr;
        Status state = runtime_->setKernel(&broadcast_kernel, "broadcast", precision);
        CHECK_EXPR_RETURN_FAILURE(state == Status::SUCCESS, "CLTensor::broadCast() setKernel fail");
        state = runtime_->setKernelArg(broadcast_kernel.get(),
                                       from_mem,
                                       to_mem,
                                       expanded_to_dims[3],
                                       expanded_from_dims[0],
                                       expanded_from_dims[1],
                                       expanded_from_dims[2],
                                       expanded_from_dims[3]);
        CHECK_EXPR_RETURN_FAILURE(state == Status::SUCCESS, "CLTensor::broadCast() setKernelArg fail");
        state = runtime_->enqueueKernel(broadcast_kernel.get(), 3, global, local);
        CHECK_EXPR_RETURN_FAILURE(state == Status::SUCCESS, "CLTensor::broadCast() enqueueKernel fail");
    } else {
        size_t global = output->getTotalSizeFromDims();
        std::shared_ptr<struct _cl_kernel> ndbroadcast_kernel = nullptr;
        Status state = runtime_->setKernel(&ndbroadcast_kernel, "ndbroadcast", precision);
        CHECK_EXPR_RETURN_FAILURE(state == Status::SUCCESS, "CLTensor::broadCastTo() setKernel fail");

        // wrap uint32_t* in_dims, out_dims into cl_mem
        const size_t dims_bytes = (size_t)(output->getNumOfDims()) * sizeof(uint32_t);
        std::shared_ptr<CLBuffer> in_dims_buffer = std::make_shared<CLBuffer>(dims_bytes);
        in_dims_buffer->assignBuffer(runtime_->allocBuffer(dims_bytes, false));
        std::shared_ptr<CLBuffer> out_dims_buffer = std::make_shared<CLBuffer>(dims_bytes);
        out_dims_buffer->assignBuffer(runtime_->allocBuffer(dims_bytes, false));
        if (output->getNumOfDims() != getNumOfDims()) {
            NDims expanded_in_dims(output->getNumOfDims() - getNumOfDims(), 1);
            expanded_in_dims.insert(expanded_in_dims.end(), vdims_.begin(), vdims_.end());
            state = runtime_->writeBuffer(
                in_dims_buffer->getDataPtr(), expanded_in_dims.data(), sizeof(uint32_t), expanded_in_dims.size(), true);
            CHECK_EXPR_RETURN_FAILURE(state == Status::SUCCESS, "runtime_->writeBuffer() fail");
        } else {
            state =
                runtime_->writeBuffer(in_dims_buffer->getDataPtr(), vdims_.data(), sizeof(uint32_t), vdims_.size(), true);
            CHECK_EXPR_RETURN_FAILURE(state == Status::SUCCESS, "runtime_->writeBuffer() fail");
        }
        state = runtime_->writeBuffer(out_dims_buffer->getDataPtr(),
                                      const_cast<NDims &>(output->getDims()).data(),
                                      sizeof(uint32_t),
                                      output->getNumOfDims(),
                                      true);
        CHECK_EXPR_RETURN_FAILURE(state == Status::SUCCESS, "runtime_->writeBuffer() fail");

        // broadcast with expanded in_dims, e.g. [A] -> [1, 1, 1, A]
        state = runtime_->setKernelArg(ndbroadcast_kernel.get(),
                                       buf_->getDataPtr(),
                                       output->getDataPtr(),
                                       in_dims_buffer->getDataPtr(),
                                       out_dims_buffer->getDataPtr(),
                                       output->getNumOfDims(),
                                       getTotalSizeFromDims());
        CHECK_EXPR_RETURN_FAILURE(state == Status::SUCCESS, "CLTensor::broadCastTo() setKernelArg fail");
        state = runtime_->enqueueKernel(ndbroadcast_kernel.get(), (cl_uint)1, &global, NULL);
        CHECK_EXPR_RETURN_FAILURE(state == Status::SUCCESS, "CLTensor::broadCastTo() enqueueKernel fail");
    }
    return Status::SUCCESS;
}

Status CLTensor::broadCastTexture2d(const cl_mem &in_mem, Dim4 dim) {
    uint32_t depth = IntegralDivideRoundUp(dim.c, 4);
    std::shared_ptr<struct _cl_kernel> broadcast_texture2d_kernel = nullptr;
    Status state;

    state = runtime_->setKernel(&broadcast_texture2d_kernel, "broadcast_texture2d", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    Status status = runtime_->setKernelArg(
        broadcast_texture2d_kernel.get(), in_mem, buf_->getDataPtr(), dims_.n, dims_.c, dims_.h, dims_.w, depth);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t local[3] = {16, 8, 1};
    size_t global[3] = {0};

    global[0] = alignTo(dim.w, local[0]);
    global[1] = alignTo(dim.h, local[1]);
    global[2] = alignTo(depth, local[2]);

    status = runtime_->enqueueKernel(broadcast_texture2d_kernel.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");
    return Status::SUCCESS;
}

/**
 * @brief [Android NN-exclusive] For N-D Operations (ADD, MUL, DIV, SUB...) wrapped in NchwBlock,
 * their 1-D, 2-D and 3-D input should be converted to NCHW layout before broadcasted to a 4-D
 * tensor.
 * @details
 */

Status CLTensor::nchwBlockBroadCastTo(std::shared_ptr<CLTensor> output) {
    DEBUG_PRINT("CLTensor::nchwBlockBroadCastTo() is called");
    if (getNumOfDims() < 4 && output->getNumOfDims() == 4) {
        const NDims ori_dims = vdims_;
        NDims from_dims_nchw;
        bool status = reorder123DimsTo4DimsForBroadcast(ori_dims, from_dims_nchw);
        CHECK_EXPR_RETURN_FAILURE(status, "reorder123DimsTo4DimsForBroadcast() failed.");
        if (getNumOfDims() == 1) {
            reconfigureDims(from_dims_nchw);
            Status state = broadCastTo(output);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "from_tensor_nchw->broadCastTo() failed.");
            reconfigureDims(ori_dims);
        } else {
            std::shared_ptr<CLTensor> from_tensor_nchw = std::make_shared<CLTensor>(runtime_,
                                                                                    precision_,
                                                                                    getDataType(),
                                                                                    from_dims_nchw,
                                                                                    getDataOrder(),
                                                                                    1.0,
                                                                                    0,
                                                                                    BufferType::DEDICATED,
                                                                                    StorageType::BUFFER,
                                                                                    get_buffer_type(),
                                                                                    get_buffer_index());
            Status state = convertToNCHW(from_tensor_nchw);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW() failed.");
            state = from_tensor_nchw->broadCastTo(output);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "from_tensor_nchw->broadCastTo() failed.");
        }
        return Status::SUCCESS;
    }
    return broadCastTo(output);
}

Status CLTensor::convertInt2Float() {
    DEBUG_PRINT("%s(+) is called", __func__);
    auto num = getTotalSizeFromDims();
    auto bytes = getNumOfBytes();
    std::shared_ptr<CLBuffer> middle_buffer = std::make_shared<CLBuffer>(bytes);
    middle_buffer->assignBuffer(runtime_->allocBuffer(bytes, false));
    runtime_->copyInt2Float(middle_buffer->getDataPtr(), buf_->getDataPtr(), num);
    size_t offset = 0;
    runtime_->copyBuffer(buf_->getDataPtr(), middle_buffer->getDataPtr(), offset, offset, bytes);
    runtime_->releaseBuffer(middle_buffer);
    DEBUG_PRINT("%s(-) is called", __func__);
    return Status::SUCCESS;
}

Status CLTensor::convertFloat2Int() {
    DEBUG_PRINT("%s(+) is called", __func__);
    auto num = getTotalSizeFromDims();
    auto bytes = getNumOfBytes();
    std::shared_ptr<CLBuffer> middle_buffer = std::make_shared<CLBuffer>(bytes);
    middle_buffer->assignBuffer(runtime_->allocBuffer(bytes, false));
    runtime_->copyFloat2Int(middle_buffer->getDataPtr(), buf_->getDataPtr(), num);
    size_t offset = 0;
    runtime_->copyBuffer(buf_->getDataPtr(), middle_buffer->getDataPtr(), offset, offset, bytes);
    runtime_->releaseBuffer(middle_buffer);
    DEBUG_PRINT("%s(-) is called", __func__);
    return Status::SUCCESS;
}
void CLTensor::print(std::string name) {
    if (ENABLECLTENSORPRINT) {
        ITensor::placeholder(name + " BEGIN", ColorType::WHITE, true);
        // auto tmp_tensor = std::static_pointer_cast<CLTensor>(tensor);
        auto tmp_tensor = this;
        printf("input dim %d %d %d %d\n",
               tmp_tensor->getDim().n,
               tmp_tensor->getDim().c,
               tmp_tensor->getDim().h,
               tmp_tensor->getDim().w);
        printf(" scale  %f ZeroPoint %d getDataType %d\n",
               tmp_tensor->getScale(),
               tmp_tensor->getZeroPoint(),
               tmp_tensor->getDataType());

        std::string golden_txtdata = "/data/local/tmp/zhanglt_test/eden/" + std::to_string(tmp_tensor->getDim().n) + "_" +
                                     std::to_string(tmp_tensor->getDim().c) + "_" + std::to_string(tmp_tensor->getDim().h) +
                                     "_" + std::to_string(tmp_tensor->getDim().w) + ".txt";

        printf("path %s\n", golden_txtdata.c_str());

        tmp_tensor->dumpTensorData(golden_txtdata.c_str());
        ITensor::placeholder(name + " END");
    }
}
}  // namespace gpu
}  // namespace ud
}  // namespace enn
