#include "CFUInverter.hpp"

namespace enn {
namespace ud {
namespace cpu {

CFUInverter::CFUInverter(const PrecisionType& precision, const std::string soc_name) {
    precision_ = precision;
    soc_name_ = soc_name;
    data_type = DataType::UNKNOWN;
    width = 0;
    height = 0;
    channel = 0;
    cols_in_cell = 0;
    lines_in_cell = 0;
    interleaved_slices = 0;
    idps = 0;
    unit_size = 0;
    output_size = 0;
}

Status CFUInverter::initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                               const int32_t& channel, const int32_t& cols_in_cell, const int32_t& lines_in_cell,
                               const int32_t& interleaved_slices, const int32_t& idps, const int32_t& unit_size) {
    ENN_UNUSED(input);
    this->width = width;
    this->height = height;
    this->channel = channel;
    this->cols_in_cell = cols_in_cell;
    this->lines_in_cell = lines_in_cell;
    this->interleaved_slices = interleaved_slices;
    this->idps = idps;
    this->unit_size = unit_size;
    return Status::SUCCESS;
}

template <typename T>
Status CFUInverter::executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor,
                                  std::shared_ptr<NEONTensor<T>> output_tensor) {
    T* input_data = input_tensor->getBufferPtr();
    T* output_data = output_tensor->getBufferPtr();

    output_size = get_output_size(output_tensor, data_type);

    DEBUG_PRINT(
        "width[%d], height[%d], channel[%d], cols_in_cell[%d], lines_in_cell[%d], interleaved_slices[%d], output_size[%d]\n",
        width, height, channel, cols_in_cell, lines_in_cell, interleaved_slices, output_size);

    if (soc_name_ == enn::platform::EXYNOS2100 || soc_name_ == enn::platform::EXYNOS991 ||
        soc_name_ == enn::platform::EXYNOS9840 || soc_name_ == enn::platform::EXYNOS9815 ||
        soc_name_ == enn::platform::EXYNOS9925
#ifdef VELOCE_SOC
        || soc_name_ == enn::platform::VELOCE_EXYNOS991 || soc_name_ == enn::platform::VELOCE_EXYNOS9925
#endif
    ) {
        if (data_type == DataType::INT16) {
            return executeKernel_impl_common((int8_t*)input_data, (int8_t*)output_data, width, height, channel, cols_in_cell,
                                             lines_in_cell, interleaved_slices);
        } else {
            return executeKernel_impl_common(input_data, output_data, width, height, channel, cols_in_cell, lines_in_cell,
                                             interleaved_slices);
        }
    } else {
        if (data_type == DataType::INT16) {
            return executeKernel_impl_KITT((int8_t*)input_data, output_data, width, height, channel, cols_in_cell,
                                           lines_in_cell, interleaved_slices, idps, unit_size);
        } else {
            return executeKernel_impl_KITT(input_data, output_data, width, height, channel, cols_in_cell, lines_in_cell,
                                           interleaved_slices, idps, unit_size);
        }
    }
}

template <typename T>
Status CFUInverter::executeKernel_impl_common(T* src_addr, T* dest_addr, int32_t width, int32_t height, int32_t channel,
                                              int32_t cols_in_cell, int32_t lines_in_cell, int32_t interleaved_slices) {
    DEBUG_PRINT("called\n");

    int cell_size = cols_in_cell * lines_in_cell;
    if (data_type == DataType::HALF) {
        cell_size *= (interleaved_slices /= sizeof(_Float16_t));
    } else {
        cell_size *= interleaved_slices;
    }
    int frame_size = height * width;
    int slice, num_slices = channel / cell_size, partial_slice = channel % cell_size;
    int counter = 0, source_index = 0, total_slices = (partial_slice != 0) ? num_slices + 1 : num_slices;
    // iterating through the bytes in cell formatted data
    for (slice = 0; slice <= num_slices; ++slice) {
        if (slice == num_slices) {
            cell_size = partial_slice;
        }

        for (int cs = 0; cs < cell_size; ++cs) {
            for (int f = 0; f < frame_size; ++f) {
                // calculating the offset from starting pointer
                // storing the output in correct sequence one byte at a time
                source_index = cs + f * interleaved_slices + slice * frame_size * interleaved_slices;
                if (counter < output_size) {
                    dest_addr[counter] = src_addr[source_index];
                } else {
                    DEBUG_PRINT("Skip out of range dest_addr[%d]\n", counter);
                }
                counter++;

                if (data_type == DataType::INT16) {
                    dest_addr[counter] = src_addr[source_index + frame_size * interleaved_slices * total_slices];
                    counter++;
                }
            }
        }
    }

    return Status::SUCCESS;
}

template <typename T1, typename T2>
Status CFUInverter::executeKernel_impl_KITT(T1* src_addr, T2* dest_addr, int32_t width, int32_t height, int32_t channel,
                                            int32_t cols_in_cell, int32_t lines_in_cell, int32_t interleaved_slices,
                                            int32_t idps, int32_t unit_size) {
    DEBUG_PRINT("called\n");

    Arguments args(width, height, channel, cols_in_cell, lines_in_cell, interleaved_slices, idps, unit_size);
    Params p(args);
    const int cell_size = cols_in_cell * lines_in_cell;
    uint32_t outIdx = 0;

    for (std::size_t IdpInd = 0; IdpInd < p.idps; IdpInd++) {
        auto IdpFstSlice = IdpInd * p.idpBaseSlices;
        auto ValidIdpSlices = std::min(args.slices - IdpFstSlice, p.idpBaseSlices);

        for (std::size_t FstSliceGrInd = 0; FstSliceGrInd < p.idpSlices; FstSliceGrInd += args.interleaved_slices) {
            auto ValidSlices =
                std::min((FstSliceGrInd >= ValidIdpSlices) ? 0 : ValidIdpSlices - FstSliceGrInd, args.interleaved_slices);

            if (data_type == DataType::INT16) {
                for (std::size_t SliceInd = 0; SliceInd < ValidSlices; SliceInd++) {
                    auto InBufSliceOfs = args.height * args.width * (SliceInd + IdpFstSlice);

                    for (size_t FstLineStripeInd = 0; FstLineStripeInd < p.npuLinesSize;
                         FstLineStripeInd += args.lines_in_cell) {
                        for (size_t FstColStripeInd = 0; FstColStripeInd < p.npuColsSize;
                             FstColStripeInd += args.cols_in_cell) {
                            for (size_t LineInd = FstLineStripeInd; LineInd < FstLineStripeInd + args.lines_in_cell;
                                 LineInd++) {
                                for (size_t ColInd = FstColStripeInd; ColInd < FstColStripeInd + args.cols_in_cell;
                                     ColInd++) {
                                    dest_addr[InBufSliceOfs + LineInd * width + ColInd] =
                                        (src_addr[outIdx] & 0xFF) | ((src_addr[outIdx + 32] & 0xFF) << 8);
                                    outIdx++;
                                }
                            }
                            outIdx += cell_size;
                        }
                    }
                }
            } else {
                auto InBufSliceGrOfs = p.npuLinesSize * p.npuColsSize * (p.idpSlices * IdpInd + FstSliceGrInd);

                for (std::size_t SliceInd = 0; SliceInd < ValidSlices; SliceInd++) {
                    auto InBufSliceOfs = InBufSliceGrOfs + args.lines_in_cell * args.cols_in_cell * SliceInd;

                    for (std::size_t FstLineStripeInd = 0; FstLineStripeInd < p.npuLinesSize;
                         FstLineStripeInd += args.lines_in_cell) {
                        auto ValidLines = std::min(args.height - FstLineStripeInd, args.lines_in_cell);
                        auto LinesStrideOfs = InBufSliceOfs + FstLineStripeInd * p.npuColsSize * args.interleaved_slices;

                        for (std::size_t LineInd = 0; LineInd < ValidLines; LineInd++) {
                            for (std::size_t FstColStripeInd = 0; FstColStripeInd < p.npuColsSize;
                                 FstColStripeInd += args.cols_in_cell) {
                                auto ColsStrideOfs =
                                    LinesStrideOfs + FstColStripeInd * args.lines_in_cell * args.interleaved_slices;
                                auto ValidCols = std::min(args.width - FstColStripeInd, args.cols_in_cell);

                                for (std::size_t ColInd = 0; ColInd < ValidCols; ColInd++)
                                    dest_addr[outIdx++] = src_addr[ColsStrideOfs + LineInd * args.cols_in_cell + ColInd];
                            }
                        }
                    }
                }
            }
        }
    }

    return Status::SUCCESS;
}

Status CFUInverter::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    data_type = input->getDataType();

    switch (data_type) {
        case DataType::FLOAT16: {
            DEBUG_PRINT("DataType::FLOAT16\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<_Float16_t>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<_Float16_t>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        case DataType::INT16: {
            DEBUG_PRINT("DataType::INT16\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<int16_t>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<int16_t>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        case DataType::INT8:
        case DataType::UINT8: {
            DEBUG_PRINT("DataType::INT8 or DataType::UINT8\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<int8_t>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<int8_t>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        default: {
            ERROR_PRINT("Data type is not supported\n");
            return Status::FAILURE;
        }
    }
}

Status CFUInverter::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
