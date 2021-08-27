#include "NEONCFUConverter.hpp"

namespace enn {
namespace ud {
namespace cpu {

NEONCFUConverter::NEONCFUConverter(const PrecisionType& precision, const std::string soc_name) {
    precision_ = precision;
    soc_name_ = soc_name;
    data_type = DataType::UNKNOWN;
    width = 0;
    height = 0;
    channel = 0;
    cols_in_cell = 0;
    lines_in_cell = 0;
    interleaved_slices = 0;
    pad_value = 0;
}

Status NEONCFUConverter::initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                                    const int32_t& channel, const int32_t& cols_in_cell, const int32_t& lines_in_cell,
                                    const int32_t& interleaved_slices, const int32_t& pad_value) {
    UNUSED(input);
    this->width = width;
    this->height = height;
    this->channel = channel;
    this->cols_in_cell = cols_in_cell;
    this->lines_in_cell = lines_in_cell;
    this->interleaved_slices = interleaved_slices;
    this->pad_value = pad_value;
    return Status::SUCCESS;
}

template <typename T>
Status NEONCFUConverter::executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor,
                                       std::shared_ptr<NEONTensor<T>> output_tensor) {
    T* input_data = input_tensor->getBufferPtr();
    T* output_data = output_tensor->getBufferPtr();

    DEBUG_PRINT("width[%d], height[%d], channel[%d], pad_value[%d]\n", width, height, channel, pad_value);

    if (soc_name_ == enn::platform::EXYNOS2100 || soc_name_ == enn::platform::EXYNOS991 ||
        soc_name_ == enn::platform::EXYNOS9840 || soc_name_ == enn::platform::EXYNOS9815 ||
        soc_name_ == enn::platform::EXYNOS9925
#ifdef VELOCE_SOC
        || soc_name_ == enn::platform::VELOCE_EXYNOS991 || soc_name_ == enn::platform::VELOCE_EXYNOS9925
#endif
    ) {
        if (data_type == DataType::INT16) {
            return executeKernel_impl_common((int8_t*)input_data, (int8_t*)output_data, width, height, channel, pad_value);
        } else {
            return executeKernel_impl_common(input_data, output_data, width, height, channel, pad_value);
        }
    } else if (soc_name_ == enn::platform::HR80 || soc_name_ == enn::platform::EXYNOSAUTO9) {
        return executeKernel_impl_KITT(input_data, output_data, width, height, channel, pad_value);
    } else {
        DEBUG_PRINT("Unknown SoC. It will work the same way as E2100\n");
        return executeKernel_impl_common(input_data, output_data, width, height, channel, pad_value);
    }
}

template <typename T>
Status NEONCFUConverter::executeKernel_impl_common(T* src_addr, T* dest_addr, int32_t width, int32_t height, int32_t channel,
                                                   int32_t pad_value) {
    DEBUG_PRINT("called\n");

    int cell_size = cols_in_cell * lines_in_cell * interleaved_slices;
    int source_index = 0;
    int counter = 0;
    int slice, num_slices = channel / cell_size, partial_slice = channel % cell_size;
    int frame_size = height * width;
    int byte_size = 1;

    if (data_type == DataType::INT16) {
        byte_size = 2;
    }

    for (int byte = 0; byte < byte_size; ++byte) {
        int cs_iter = cell_size;
        // assuming CHW format of data with C being the outermost dimension
        for (slice = 0; slice <= num_slices; ++slice) {
            if (slice == num_slices) {
                cs_iter = partial_slice;
            }
            for (int f = 0; f < frame_size; ++f) {
                for (int cs = 0; cs < cs_iter; ++cs) {
                    source_index = (f + frame_size * cs + frame_size * slice * cell_size);
                    dest_addr[counter] = src_addr[source_index * byte_size + byte];
                    counter++;
                }
                // if partial_slice is 0, it isn't required to clear padding
                // because counter is same with dest_addr's length.
                if (partial_slice && slice == num_slices) {
                    if (data_type == DataType::FLOAT16) {
                        memset(dest_addr + counter, 0, (interleaved_slices - partial_slice) * sizeof(_Float16_t));
                    } else {
                        memset(dest_addr + counter, pad_value, interleaved_slices - partial_slice);
                    }
                    counter += (interleaved_slices - partial_slice);
                }
            }
        }
    }

    return Status::SUCCESS;
}

Status NEONCFUConverter::executeKernel_impl_KITT(int16_t* src_addr, int16_t* dest_addr, int32_t width, int32_t height,
                                                 int32_t channel, int32_t pad_value) {
    DEBUG_PRINT("called\n");

    int out_index = 0;
    int unit_size = 2;
    int idps = 4;
    int idp_fst_slice = 0, idp_base_slices = 0, idp_slices = 0, idp_idx = 0;
    int fst_slice_gr_idx = 0, fst_line_stripe_idx = 0, fst_col_stripe_idx = 0, slice_idx = 0, line_idx = 0, col_idx = 0;
    int valid_idp_slices = 0, valid_slices = 0, valid_lines = 0, valid_cols = 0, inbuf_slice_ofs = 0;
    int npu_cols_size = 0, npu_lines_size = 0;
    // Columns = width, Lines = height
    int cols_in_cell = 16, lines_in_cell = 8;
    int interleaved_slices = 2;

    npu_cols_size = (width + cols_in_cell - 1) / cols_in_cell * cols_in_cell;
    npu_lines_size = (height + lines_in_cell - 1) / lines_in_cell * lines_in_cell;

    if (idps * unit_size > channel + unit_size - 1)
        idps = (channel + unit_size - 1) / unit_size;
    else
        idps = 4;

    idp_base_slices = (channel + idps * unit_size - 1) / (idps * unit_size) * unit_size;
    idp_slices = (idp_base_slices + interleaved_slices - 1) / interleaved_slices * interleaved_slices;

    DEBUG_PRINT("npu_cols_size=[%d], npu_lines_size=[%d]\n", npu_cols_size, npu_lines_size);
    DEBUG_PRINT("idps=[%d], idp_base_slices=[%d], idp_slices=[%d]\n", idps, idp_base_slices, idp_slices);
    DEBUG_PRINT("width=[%d], height=[%d], channel=[%d]\n", width, height, channel);

    int8_t* pSrc = (int8_t*)src_addr;
    int8_t* pDest = (int8_t*)dest_addr;
    int32_t flag = 0;
    int32_t cell_size = lines_in_cell * cols_in_cell;

    for (idp_idx = 0; idp_idx < idps; idp_idx++) {
        idp_fst_slice = idp_idx * idp_base_slices;
        valid_idp_slices = std::min(channel - idp_fst_slice, idp_base_slices);
        for (fst_slice_gr_idx = 0; fst_slice_gr_idx < idp_slices; fst_slice_gr_idx += interleaved_slices) {
            valid_slices = std::min(valid_idp_slices - fst_slice_gr_idx, interleaved_slices);
            // If number of channels is not divided by num idp (usually 4), last idp will contain
            // less number of slices than others and in this case valid_slices will be negative.
            // Catch this now.
            if (valid_slices < 0) {
                valid_slices = 0;
            }
            for (slice_idx = fst_slice_gr_idx; slice_idx < fst_slice_gr_idx + valid_slices; slice_idx++) {
                inbuf_slice_ofs = (2 * width) * height * (idp_fst_slice + slice_idx);
                for (fst_line_stripe_idx = 0; fst_line_stripe_idx < npu_lines_size; fst_line_stripe_idx += lines_in_cell) {
                    valid_lines = std::min(height - fst_line_stripe_idx, lines_in_cell);
                    for (fst_col_stripe_idx = 0; fst_col_stripe_idx < npu_cols_size; fst_col_stripe_idx += cols_in_cell) {
                        valid_cols = std::min(width - fst_col_stripe_idx, cols_in_cell);
                        for (line_idx = fst_line_stripe_idx; line_idx < fst_line_stripe_idx + valid_lines; line_idx++) {
#ifndef NEON_OPT
                            for (col_idx = fst_col_stripe_idx; col_idx < fst_col_stripe_idx + valid_cols; col_idx++) {
                                pDest[out_index] = pSrc[inbuf_slice_ofs + line_idx * (2 * width) + (col_idx * 2)];
                                pDest[out_index + cell_size] =
                                    pSrc[inbuf_slice_ofs + line_idx * (2 * width) + ((col_idx * 2) + 1)];
                                out_index++;
                            }
#else
                            if (valid_cols < 16) {
                                for (col_idx = fst_col_stripe_idx; col_idx < fst_col_stripe_idx + valid_cols; col_idx++) {
                                    pDest[out_index] = pSrc[inbuf_slice_ofs + line_idx * (2 * width) + (col_idx * 2)];
                                    pDest[out_index + cell_size] =
                                        pSrc[inbuf_slice_ofs + line_idx * (2 * width) + ((col_idx * 2) + 1)];
                                    out_index++;
                                }
                            } else {
                                int8x16x2_t input =
                                    vld2q_s8(pSrc + inbuf_slice_ofs + line_idx * (2 * width) + (col_idx * 2));
                                vst1q_s8(pDest + out_index, input.val[0]);
                                vst1q_s8(pDest + out_index + cell_size, input.val[1]);
                                out_index += valid_cols;
                            }
#endif
                            for (col_idx = fst_col_stripe_idx + valid_cols; col_idx < fst_col_stripe_idx + cols_in_cell;
                                 col_idx++) {
                                pDest[out_index] = pad_value;
                                pDest[out_index + cell_size] = pad_value;
                                out_index++;
                            }
                            flag = 1;
                        }
                        for (line_idx = fst_line_stripe_idx + valid_lines; line_idx < fst_line_stripe_idx + lines_in_cell;
                             line_idx++) {
#ifndef NEON_OPT
                            for (col_idx = 0; col_idx < cols_in_cell; col_idx++) {
                                pDest[out_index] = pad_value;
                                pDest[out_index + cell_size] = pad_value;
                                out_index++;
                            }
#else
                            vst1q_s8(pDest + out_index, vmovq_n_s8(0));
                            vst1q_s8(pDest + out_index + cell_size, vmovq_n_s8(0));
                            out_index += cols_in_cell;
#endif
                            flag = 1;
                        }
                        if (flag == 1) {
                            out_index += cell_size;
                            flag = 0;
                        }
                    }
                }
            }
        }
    }

    return Status::SUCCESS;
}

Status NEONCFUConverter::executeKernel_impl_KITT(int8_t* src_addr, int8_t* dest_addr, int32_t width, int32_t height,
                                                 int32_t channel, int32_t pad_value) {
    DEBUG_PRINT("called\n");
    UNUSED(pad_value);

    int out_index = 0;
    int unit_size = 2;
    int idps = 4;
    int idp_fst_slice = 0, idp_base_slices = 0, idp_slices = 0, idp_idx = 0;
    int fst_slice_gr_idx = 0, fst_line_stripe_idx = 0, fst_col_stripe_idx = 0, slice_idx = 0, line_idx = 0, col_idx = 0;
    int valid_idp_slices = 0, valid_slices = 0, valid_lines = 0, valid_cols = 0, inbuf_slice_ofs = 0;
    int npu_cols_size = 0, npu_lines_size = 0;
    // Columns = width, Lines = height
    int cols_in_cell = 16, lines_in_cell = 8;
    int interleaved_slices = 2;

    npu_cols_size = (width + cols_in_cell - 1) / cols_in_cell * cols_in_cell;
    npu_lines_size = (height + lines_in_cell - 1) / lines_in_cell * lines_in_cell;

    if (idps * unit_size > channel + unit_size - 1)
        idps = (channel + unit_size - 1) / unit_size;
    else
        idps = 4;

    idp_base_slices = (channel + idps * unit_size - 1) / (idps * unit_size) * unit_size;
    idp_slices = (idp_base_slices + interleaved_slices - 1) / interleaved_slices * interleaved_slices;

    DEBUG_PRINT("npu_cols_size=[%d], npu_lines_size=[%d]\n", npu_cols_size, npu_lines_size);
    DEBUG_PRINT("idps=[%d], idp_base_slices=[%d], idp_slices=[%d]\n", idps, idp_base_slices, idp_slices);
    DEBUG_PRINT("width=[%d], height=[%d], channel=[%d]\n", width, height, channel);

    for (idp_idx = 0; idp_idx < idps; idp_idx++) {
        idp_fst_slice = idp_idx * idp_base_slices;
        valid_idp_slices = std::min(channel - idp_fst_slice, idp_base_slices);
        for (fst_slice_gr_idx = 0; fst_slice_gr_idx < idp_slices; fst_slice_gr_idx += interleaved_slices) {
            valid_slices = std::min(valid_idp_slices - fst_slice_gr_idx, interleaved_slices);
            // If number of channels is not divided by num idp (usually 4), last idp will contain
            // less number of slices than others and in this case valid_slices will be negative.
            // Catch this now.
            if (valid_slices < 0) {
                valid_slices = 0;
            }
            for (fst_line_stripe_idx = 0; fst_line_stripe_idx < npu_lines_size; fst_line_stripe_idx += lines_in_cell) {
                valid_lines = std::min(height - fst_line_stripe_idx, lines_in_cell);
                for (fst_col_stripe_idx = 0; fst_col_stripe_idx < npu_cols_size; fst_col_stripe_idx += cols_in_cell) {
                    valid_cols = std::min(width - fst_col_stripe_idx, cols_in_cell);
                    for (slice_idx = fst_slice_gr_idx; slice_idx < fst_slice_gr_idx + valid_slices; slice_idx++) {
                        inbuf_slice_ofs = width * height * (idp_fst_slice + slice_idx);
                        for (line_idx = fst_line_stripe_idx; line_idx < fst_line_stripe_idx + valid_lines; line_idx++) {
#ifndef NEON_OPT
                            for (col_idx = fst_col_stripe_idx; col_idx < fst_col_stripe_idx + valid_cols; col_idx++) {
                                dest_addr[out_index++] = src_addr[inbuf_slice_ofs + line_idx * width + col_idx];
                            }
#else
                            int8x16_t input = vld1q_s8(src_addr + inbuf_slice_ofs + line_idx * width + fst_col_stripe_idx);
                            vst1q_s8(dest_addr + out_index, input);
                            out_index += valid_cols;
#endif
                            for (col_idx = fst_col_stripe_idx + valid_cols; col_idx < fst_col_stripe_idx + cols_in_cell;
                                 col_idx++) {
                                dest_addr[out_index++] = 0;
                            }
                        }
                        for (line_idx = fst_line_stripe_idx + valid_lines; line_idx < fst_line_stripe_idx + lines_in_cell;
                             line_idx++) {
#ifndef NEON_OPT
                            for (col_idx = 0; col_idx < cols_in_cell; col_idx++)
                                dest_addr[out_index++] = 0;
#else
                            vst1q_s8(dest_addr + out_index, vmovq_n_s8(0));
                            out_index += cols_in_cell;
#endif
                        }
                    }
                    for (slice_idx = fst_slice_gr_idx + valid_slices; slice_idx < fst_slice_gr_idx + interleaved_slices;
                         slice_idx++) {
                        for (line_idx = 0; line_idx < lines_in_cell; line_idx++) {
#ifndef NEON_OPT
                            for (col_idx = 0; col_idx < cols_in_cell; col_idx++)
                                dest_addr[out_index++] = 0;
#else
                            vst1q_s8(dest_addr + out_index, vmovq_n_s8(0));
                            out_index += cols_in_cell;
#endif
                        }
                    }
                }
            }
        }
    }

    return Status::SUCCESS;
}

Status NEONCFUConverter::execute(std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
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

Status NEONCFUConverter::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
