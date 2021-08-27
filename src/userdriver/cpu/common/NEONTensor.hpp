#pragma once

#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

template <typename T>
class NEONTensor : public ITensor {
public:
    NEONTensor(const Dim4& dims, const PrecisionType& precision, const uint32_t& buffer_type = 0,
               const int32_t& buffer_index = UNDEFINED, const float& scale = 1.0, const int32_t& offset = 0)
        : dims_(dims), precision_(precision), scale_(scale), offset_(offset) {
        vdims_ = {dims.n, dims.c, dims.h, dims.w};
        buf_.reset(new T[getDimensionsSize(vdims_)]());
        order_ = DataOrder::NCHW;
        data_type_ = data_type_map_[typeid(T).name()];
        buffer_type_ = buffer_type;
        buffer_index_ = buffer_index;
    }

    NEONTensor(const NDims& ndim, const PrecisionType& precision, const uint32_t& buffer_type = 0,
               const int32_t& buffer_index = UNDEFINED, const float& scale = 1.0, const int32_t& offset = 0)
        : vdims_(ndim), precision_(precision), scale_(scale), offset_(offset) {
        dims_ = extendToDim4(vdims_);
        buf_.reset(new T[getDimensionsSize(vdims_)]());
        order_ = DataOrder::NCHW;
        data_type_ = data_type_map_[typeid(T).name()];
        buffer_type_ = buffer_type;
        buffer_index_ = buffer_index;
    }

    NEONTensor(T* data, const Dim4& dims, const PrecisionType& precision, const uint32_t& buffer_type = 0,
               const int32_t& buffer_index = UNDEFINED, const float& scale = 1.0, const int32_t& offset = 0)
        : dims_(dims), precision_(precision), scale_(scale), offset_(offset) {
        vdims_ = {dims.n, dims.c, dims.h, dims.w};
        buf_.reset(new T[getDimensionsSize(vdims_)]());
        std::copy(data, data + getDimensionsSize(vdims_), buf_.get());
        order_ = DataOrder::NCHW;
        data_type_ = data_type_map_[typeid(T).name()];
        buffer_type_ = buffer_type;
        buffer_index_ = buffer_index;
    }

    NEONTensor(T* data, const NDims& ndim, const PrecisionType& precision, const uint32_t& buffer_type = 0,
               const int32_t& buffer_index = UNDEFINED, const float& scale = 1.0, const int32_t& offset = 0)
        : vdims_(ndim), precision_(precision), scale_(scale), offset_(offset) {
        dims_ = extendToDim4(vdims_);
        buf_.reset(new T[getDimensionsSize(vdims_)]());
        std::copy(data, data + getDimensionsSize(vdims_), buf_.get());
        order_ = DataOrder::NCHW;
        data_type_ = data_type_map_[typeid(T).name()];
        buffer_type_ = buffer_type;
        buffer_index_ = buffer_index;
    }

    size_t getNumOfBytes() override {
        return (size_t)getTotalSizeFromDims() * sizeof(T);
    }

    DataType getDataType() override {
        return data_type_;
    }

    PrecisionType getPrecisionType() override {
        return precision_;
    }

    const Dim4& getDim() override {
        return dims_;
    }

    const NDims& getDims() override {
        return vdims_;
    }

    uint32_t getDims(const int32_t& idx) override {
        int32_t vdims_size = static_cast<int32_t>(vdims_.size());
        if (idx > vdims_size - 1 || idx + vdims_size < 0) {
            ERROR_PRINT("Invalid idx: %d", idx);
            return 0;
        }
        int32_t m_idx = idx < 0 ? (idx + vdims_size) : idx;
        return vdims_.at(m_idx);
    }

    uint32_t getNumOfDims() override {
        return vdims_.size();
    }

    DataOrder getDataOrder() override {
        return order_;
    }

    Status readData(DataPtr data, bool = true, DataOrderChangeType type = DataOrderChangeType::OTHER,
                    void* = nullptr) override {
        UNUSED(type);
        auto total_bytes = sizeof(T) * getTotalSizeFromDims();
        memcpy(data, buf_.get(), total_bytes);
        return Status::SUCCESS;
    }
    Status writeData(DataPtr data, bool = true, DataOrderChangeType type = DataOrderChangeType::OTHER) override {
        UNUSED(type);
        auto total_bytes = sizeof(T) * getTotalSizeFromDims();
        memcpy(buf_.get(), data, total_bytes);
        return Status::SUCCESS;
    }

    Status reconfigure(const PrecisionType& precision, const DataOrder& order, const Dim4& dims, const float& scale,
                       const int32_t& offset) override {
        precision_ = precision;
        order_ = order;
        dims_ = dims;
        vdims_ = {dims.n, dims.c, dims.h, dims.w};
        scale_ = scale;
        offset_ = offset;
        buf_.reset(new T[getTotalSizeFromDims()]());
        return Status::SUCCESS;
    }

    Status reconfigure(const PrecisionType& precision, const DataOrder& order, const NDims& ndim, const float& scale,
                       const int32_t& offset) override {
        precision_ = precision;
        order_ = order;
        vdims_ = ndim;
        dims_ = extendToDim4(vdims_);
        scale_ = scale;
        offset_ = offset;
        buf_.reset(new T[getTotalSizeFromDims()]());
        return Status::SUCCESS;
    }

    Status reorder(const DataOrder& order) override {
        DEBUG_PRINT("NEONTensor::reorder() is not supported yet\n");
        order_ = order;
        return Status::FAILURE;
    }

    void queryTensorInfo() override {
        std::cout << "-------- Tensor Info -------" << std::endl;
        std::cout << "Precision: " << static_cast<std::underlying_type<PrecisionType>::type>(precision_) << std::endl;
        std::cout << "Order: " << static_cast<std::underlying_type<DataOrder>::type>(order_) << std::endl;
        std::cout << "Shape: ";
        std::cout << "(n: " << dims_.n << ") (c: " << dims_.c << ") (h: " << dims_.h << ") (w: " << dims_.w << ")" << std::endl;
        std::cout << "Offset: " << offset_ << std::endl;
        std::cout << "Scale: " << scale_ << std::endl;
        std::cout << "Buffer: " << buf_ << std::endl;
        std::cout << "----------------------------" << std::endl;
    }

    void dumpTensorData(std::string file_name) override {
        std::ofstream out_file;
        out_file.open(file_name);
        for (uint32_t i = 0; i < getTotalSizeFromDims(); i++) {
            out_file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << buf_.get()[i] << std::endl;
        }
    }

    Status reconfigureDim(const Dim4& dims) override {
        dims_ = dims;
        vdims_ = {dims.n, dims.c, dims.h, dims.w};
        return Status::SUCCESS;
    }

    Status reconfigureDims(const NDims& ndim) override {
        vdims_ = ndim;
        dims_ = extendToDim4(vdims_);
        return Status::SUCCESS;
    }

    Status reconfigureDimAndBuffer(const Dim4& dims) override {
        uint32_t size_ori = getTotalSizeFromDims();
        dims_ = dims;
        vdims_ = {dims.n, dims.c, dims.h, dims.w};
        uint32_t size_cur = getTotalSizeFromDims();
        if (size_ori != size_cur) {
            buf_.reset(new T[getTotalSizeFromDims()]());
        }
        return Status::SUCCESS;
    }

    Status reconfigureDimsAndBuffer(const NDims& ndim) override {
        uint32_t size_ori = getTotalSizeFromDims();
        vdims_ = ndim;
        dims_ = extendToDim4(vdims_);
        uint32_t size_cur = getTotalSizeFromDims();
        if (size_ori != size_cur) {
            buf_.reset(new T[getTotalSizeFromDims()]());
        }
        return Status::SUCCESS;
    }

    std::shared_ptr<T> getDataPtr() {
        return buf_;
    }

    uint32_t getDimensionsSize(NDims& ndims) {
        return std::accumulate(ndims.begin(), ndims.end(), 1u, std::multiplies<uint32_t>());
    }

    uint32_t getTotalSizeFromDims() override {
        return getDimensionsSize(vdims_);
    }

    Status zeroInit() {
        std::fill(buf_.get(), buf_.get() + getTotalSizeFromDims(), 0);
        return Status::SUCCESS;
    }

    float getScale() override {
        return scale_;
    }

    void setScale(float scale) override {
        scale_ = scale;
    }

    int32_t getZeroPoint() override {
        return offset_;
    }

    void setZeroPoint(int32_t zero_point) override {
        offset_ = zero_point;
    }

    int32_t get_buffer_index() override {
        return buffer_index_;
    }

    void set_buffer_index(int32_t index) override {
        buffer_index_ = index;
    }

    uint32_t get_buffer_type() override {
        return buffer_type_;
    }

    void set_buffer_type(uint32_t type) override {
        buffer_type_ = type;
    }

    void set_buffer_ptr(void* addr) override {
        raw_buffer = addr;
    }

    T* getBufferPtr() {
        if (raw_buffer != nullptr) {
            return (T*)raw_buffer;
        }
        return buf_.get();
    }

    Status resetInterBuffer() override {
        DEBUG_PRINT("Buffer reuse not implemented in NEON\n");
        return Status::SUCCESS;
    }

    ~NEONTensor() {
        vdims_.clear();
    }

private:
    Dim4 dims_;
    NDims vdims_;
    PrecisionType precision_;
    DataType data_type_;
    DataOrder order_;

    float scale_ = 0.0f;
    int32_t offset_ = 0;
    int32_t buffer_index_ = UNDEFINED;
    uint32_t buffer_type_ = 0;

    std::shared_ptr<T> buf_;

    std::map<std::string, DataType> data_type_map_ = {
        {typeid(float).name(),      DataType::FLOAT},
        {typeid(_Float16_t).name(), DataType::FLOAT16},
        {typeid(int32_t).name(),    DataType::INT32},
        {typeid(int8_t).name(),     DataType::INT8},
        {typeid(uint8_t).name(),    DataType::UINT8},
        {typeid(bool).name(),       DataType::BOOL},
        {typeid(int16_t).name(),    DataType::INT16},
        {typeid(uint16_t).name(),   DataType::UINT16}
    }; // default is DataType::FLOAT(0)
};  // class NEONTensor

}  // namespace cpu
}  // namespace ud
}  // namespace enn
