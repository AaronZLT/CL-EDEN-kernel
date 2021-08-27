#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/ICFUInverter.hpp"
#include "userdriver/common/op_test/test_capabilities.h"
#include "userdriver/cpu/common/NEONTensor.hpp"
#include "userdriver/cpu/common/NEONIncludes.hpp"

namespace enn {
namespace ud {
namespace cpu {

class CFUInverter : public ICFUInverter {
public:
    explicit CFUInverter(const PrecisionType& precision, const std::string soc_name);

    Status initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                      const int32_t& channel, const int32_t& cols_in_cell, const int32_t& lines_in_cell,
                      const int32_t& interleaved_slices, const int32_t& idps, const int32_t& unit_size);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

private:
    PrecisionType precision_;
    DataType data_type;

    std::string soc_name_;

    int32_t width;
    int32_t height;
    int32_t channel;
    int32_t cols_in_cell;
    int32_t lines_in_cell;
    int32_t interleaved_slices;
    int32_t idps;
    int32_t unit_size;
    int32_t output_size;

    template <typename T>
    Status executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor, std::shared_ptr<NEONTensor<T>> output_tensor);

    /**
     * @brief support exynos 2100/991/9840/9815/9925
     */
    template <typename T>
    Status executeKernel_impl_common(T* src_addr, T* dest_addr, int32_t width, int32_t height, int32_t channel,
                                     int32_t cols_in_cell, int32_t lines_in_cell, int32_t interleaved_slices);

    /**
     * @brief support others target
     */
    template <typename T1, typename T2>
    Status executeKernel_impl_KITT(T1* src_addr, T2* dest_addr, int32_t width, int32_t height, int32_t channel,
                                   int32_t cols_in_cell, int32_t lines_in_cell, int32_t interleaved_slices, int32_t idps,
                                   int32_t unit_size);

    /**
     * @brief Not support others target [FLOAT16]
     */
    Status executeKernel_impl_KITT(_Float16_t* src_addr, _Float16_t* dest_addr, int32_t width, int32_t height,
                                   int32_t channel, int32_t cols_in_cell, int32_t lines_in_cell, int32_t interleaved_slices,
                                   int32_t idps, int32_t unit_size) {
        ERROR_PRINT("Doesn't support Float16 type.\n");
        ENN_UNUSED(src_addr);
        ENN_UNUSED(dest_addr);
        ENN_UNUSED(width);
        ENN_UNUSED(height);
        ENN_UNUSED(channel);
        ENN_UNUSED(cols_in_cell);
        ENN_UNUSED(lines_in_cell);
        ENN_UNUSED(interleaved_slices);
        ENN_UNUSED(idps);
        ENN_UNUSED(unit_size);
        return Status::INVALID_PARAMS;
    }

    template <typename T>
    inline int32_t get_output_size(std::shared_ptr<NEONTensor<T>> output, DataType data_type) {
        Dim4 dims = output->getDim();
        int32_t multiple = (data_type == DataType::INT16) ? 2 : 1;
        return static_cast<int32_t>(dims.n * dims.c * dims.h * dims.w * multiple);
    }
};

struct Arguments {
    std::size_t width;
    std::size_t height;
    std::size_t lines_in_cell;
    std::size_t cols_in_cell;
    std::size_t idps;
    std::size_t unit_size;
    std::size_t slices;
    std::size_t interleaved_slices;

    explicit Arguments(const int32_t width_, const int32_t height_, const int32_t channel_, const int32_t cols_in_cell_,
                       const int32_t lines_in_cell_, const int32_t interleaved_slices_, const int32_t idps_,
                       const int32_t unit_size_) {
        width = width_;
        height = height_;
        slices = channel_;
        cols_in_cell = cols_in_cell_;
        lines_in_cell = lines_in_cell_;
        interleaved_slices = interleaved_slices_;
        idps = idps_;
        unit_size = unit_size_;
    }
};

// this struct represents `Params` from NPU_Converter.py
struct Params {
    std::size_t npuColsSize;
    std::size_t npuLinesSize;
    std::size_t idps;
    std::size_t idpBaseSlices;
    std::size_t idpSlices;

    explicit Params(const Arguments& args) {
        npuColsSize = ((args.width + args.cols_in_cell - 1) / args.cols_in_cell) * args.cols_in_cell;
        npuLinesSize = ((args.height + args.lines_in_cell - 1) / args.lines_in_cell) * args.lines_in_cell;

        if (args.idps * args.unit_size > args.slices + args.unit_size - 1) {
            idps = (args.slices + args.unit_size - 1) / args.unit_size;
        } else {
            idps = args.idps;
        }

        idpBaseSlices = (args.slices + idps * args.unit_size - 1) / (idps * args.unit_size) * args.unit_size;
        idpSlices = (idpBaseSlices + args.interleaved_slices - 1) / args.interleaved_slices * args.interleaved_slices;
    }
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn
