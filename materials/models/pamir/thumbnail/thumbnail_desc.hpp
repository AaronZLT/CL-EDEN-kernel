#ifndef __THUMBNAIL_DESC_H__
#define __THUMBNAIL_DESC_H__

#include "ofik_data.h"
#include "ofik_desc_kernel.hpp"

// Please change this to make CGO for each size

#define TC_64x64
//#define TC_64x32
//#define TC_64x16
//#define TC_32x64
//#define TC_32x32
//#define TC_32x16
//#define TC_16x16

#define USE_THREAD_GLOBAL

#ifdef TC_64x64
#define OUT_FSIZE_W 64
#define OUT_FSIZE_H 64
#ifdef USE_THREAD_GLOBAL
#define IN_TILE_W 64
#define IN_TILE_H 4
#define OUT_TILE_H 4
#define OUT_VALUE_H 16 // TILE NUMBER
#else
#define IN_TILE_W 64
#define IN_TILE_H 8
#define OUT_TILE_H 8
#define OUT_VALUE_H 8 // TILE NUMBER
#endif
#endif

#ifdef TC_64x32
#define OUT_FSIZE_W 64
#define OUT_FSIZE_H 32
#ifdef USE_THREAD_GLOBAL
#define IN_TILE_W 64
#define IN_TILE_H 2
#define OUT_TILE_H 4
#define OUT_VALUE_H 16 // TILE NUMBER
#else
#define IN_TILE_W 64
#define IN_TILE_H 4
#define OUT_TILE_H 8
#define OUT_VALUE_H 8 // TILE NUMBER
#endif
#endif


#ifdef TC_64x16
#define OUT_FSIZE_W 64
#define OUT_FSIZE_H 16
#ifdef USE_THREAD_GLOBAL
#define IN_TILE_W 64
#define IN_TILE_H 2
#define OUT_TILE_H 4
#define OUT_VALUE_H 8 // TILE NUMBER
#else
#define IN_TILE_W 64
#define IN_TILE_H 4
#define OUT_TILE_H 8
#define OUT_VALUE_H 4 // TILE NUMBER
#endif
#endif


#ifdef TC_32x64
#define OUT_FSIZE_W 32
#define OUT_FSIZE_H 64
#ifdef USE_THREAD_GLOBAL
#define IN_TILE_W 32
#define IN_TILE_H 7
#define OUT_TILE_H 7
#define OUT_VALUE_H 10 // TILE NUMBER
#else
#define IN_TILE_W 32
#define IN_TILE_H 14
#define OUT_TILE_H 14
#define OUT_VALUE_H 5 // TILE NUMBER
#endif
#endif


#ifdef TC_32x32
#define OUT_FSIZE_W 32
#define OUT_FSIZE_H 32
#ifdef USE_THREAD_GLOBAL
#define IN_TILE_W 32
#define IN_TILE_H 4
#define OUT_TILE_H 8
#define OUT_VALUE_H 8 // TILE NUMBER
#else
#define IN_TILE_W 32
#define IN_TILE_H 8
#define OUT_TILE_H 16
#define OUT_VALUE_H 4 // TILE NUMBER
#endif
#endif

#ifdef TC_32x16
#define OUT_FSIZE_W 32
#define OUT_FSIZE_H 16
#ifdef USE_THREAD_GLOBAL
#define IN_TILE_W 32
#define IN_TILE_H 2
#define OUT_TILE_H 8
#define OUT_VALUE_H 8 // TILE NUMBER
#else
#define IN_TILE_W 32
#define IN_TILE_H 8
#define OUT_TILE_H 16
#define OUT_VALUE_H 4 // TILE NUMBER
#endif
#endif


#ifdef TC_16x16
#define OUT_FSIZE_W 16
#define OUT_FSIZE_H 16
#ifdef USE_THREAD_GLOBAL
#define IN_TILE_W 16
#define IN_TILE_H 2
#define OUT_TILE_H 32
#define OUT_VALUE_H 8 // TILE NUMBER
#else
#define IN_TILE_W 16
#define IN_TILE_H 4
#define OUT_TILE_H 64
#define OUT_VALUE_H 4 // TILE NUMBER
#endif
#endif

#define OUT_TILE_W 64
#define OUT_BUF_W 64
#define OUT_BUF_H 64


#define NUM_OUTPUT_VALUES 12

#pragma PACKED_S_
typedef struct
{
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
#pragma PACKED_E_

class thumbnail_desc : public descriptor::desc_kernel
{
  public:
    thumbnail_desc() : descriptor::desc_kernel(3, 9) {}
    virtual ~thumbnail_desc() {}

    virtual void explain_kernel(descriptor::kernel_desc_t &r_desc)
    {
        r_desc.kernel_name               = "thumbnail";
#ifdef USE_THREAD_GLOBAL
        r_desc.exec_model                = OFI_THREAD_GLOBAL;
#else
        r_desc.exec_model                = OFI_THREAD_PRIVATE;
#endif
        r_desc.param_size                = sizeof(thumbnail_param_t);

        r_desc.input[0].data_type = OFI_DATA_U8;
        r_desc.input[0].tile_desc.default2d();
        r_desc.input[0].tile_desc.max_size.set2d(IN_TILE_W, IN_TILE_H * 16);
        r_desc.input[0].tile_desc.min_size.set2d(IN_TILE_W, 1);
        r_desc.input[0].logical_bank_index = 0;
        r_desc.input[1].data_type = OFI_DATA_U8;
        r_desc.input[1].tile_desc.default2d();
        r_desc.input[1].tile_desc.max_size.set2d(IN_TILE_W, IN_TILE_H * 16);
        r_desc.input[1].tile_desc.min_size.set2d(IN_TILE_W, 1);
        r_desc.input[1].logical_bank_index = 0;
        r_desc.input[2].data_type = OFI_DATA_U32;
        r_desc.input[2].tile_desc.default2d();
        r_desc.input[2].tile_desc.max_size.set2d(IN_TILE_W, IN_TILE_H);
        r_desc.input[2].tile_desc.min_size.set2d(IN_TILE_W, 1);
        r_desc.input[2].logical_bank_index = 0;

        r_desc.output[0].data_type = OFI_DATA_U32;
        r_desc.output[0].tile_desc.default2d();
        r_desc.output[0].tile_desc.max_size.set2d(OUT_TILE_W, OUT_TILE_H);
        r_desc.output[0].tile_desc.min_size.set2d(OUT_TILE_W, 1);
        r_desc.output[0].logical_bank_index = 0;
        r_desc.output[1].data_type = OFI_DATA_U32;
        r_desc.output[1].tile_desc.default2d();
        r_desc.output[1].tile_desc.max_size.set2d(OUT_TILE_W, OUT_TILE_H);
        r_desc.output[1].tile_desc.min_size.set2d(OUT_TILE_W, 1);
        r_desc.output[1].logical_bank_index = 0;
        r_desc.output[2].data_type = OFI_DATA_U32;
        r_desc.output[2].tile_desc.default2d();
        r_desc.output[2].tile_desc.max_size.set2d(OUT_TILE_W, OUT_TILE_H);
        r_desc.output[2].tile_desc.min_size.set2d(OUT_TILE_W, 1);
        r_desc.output[2].logical_bank_index = 0;
        r_desc.output[3].data_type = OFI_DATA_U32;
        r_desc.output[3].tile_desc.default2d();
        r_desc.output[3].tile_desc.max_size.set2d(OUT_TILE_W, OUT_TILE_H);
        r_desc.output[3].tile_desc.min_size.set2d(OUT_TILE_W, 1);
        r_desc.output[3].logical_bank_index = 0;
        r_desc.output[4].data_type = OFI_DATA_U32;
        r_desc.output[4].tile_desc.default2d();
        r_desc.output[4].tile_desc.max_size.set2d(OUT_TILE_W, OUT_TILE_H * 12);
        r_desc.output[4].tile_desc.min_size.set2d(OUT_TILE_W, 1);
        r_desc.output[4].logical_bank_index = 0;
        r_desc.output[5].data_type = OFI_DATA_U32;
        r_desc.output[5].tile_desc.default2d();
        r_desc.output[5].tile_desc.max_size.set2d(OUT_TILE_W, OUT_TILE_H * 6);
        r_desc.output[5].tile_desc.min_size.set2d(OUT_TILE_W, 1);
        r_desc.output[5].logical_bank_index = 0;
        r_desc.output[6].data_type = OFI_DATA_U32;
        r_desc.output[6].tile_desc.default2d();
        r_desc.output[6].tile_desc.max_size.set2d(OUT_TILE_W, OUT_TILE_H * 3);
        r_desc.output[6].tile_desc.min_size.set2d(OUT_TILE_W, 1);
        r_desc.output[6].logical_bank_index = 0;
        r_desc.output[7].data_type = OFI_DATA_U32;
        r_desc.output[7].tile_desc.default2d();
        r_desc.output[7].tile_desc.max_size.set2d(NUM_OUTPUT_VALUES, 1);
        r_desc.output[7].tile_desc.min_size.set2d(1, 1);
        r_desc.output[7].logical_bank_index = 0;
        r_desc.output[8].data_type = OFI_DATA_U32;
        r_desc.output[8].tile_desc.default2d();
        r_desc.output[8].tile_desc.max_size.set2d(IN_TILE_W, IN_TILE_H);
        r_desc.output[8].tile_desc.min_size.set2d(IN_TILE_W, 1);
        r_desc.output[8].logical_bank_index = 0;

        r_desc.vstack_size = 15000;
        //r_desc.vstack_size = 30000;
    }

    virtual void explain_tile_loop(void)
    {
        descriptor::tile_iters itile(3);
        itile[0].io_index = 0;
        itile[1].io_index = 1;
        itile[2].io_index = 2;

        descriptor::tile_iters otile(9);
        otile[0].io_index = 0;
        otile[1].io_index = 1;
        otile[2].io_index = 2;
        otile[3].io_index = 3;
        otile[4].io_index = 4;
        otile[5].io_index = 5;
        otile[6].io_index = 6;
        otile[7].io_index = 7;
        otile[8].io_index = 8;

        while_itile(itile);
        {
            tile_kernel();
            out_otile(otile);
        }
        next_itile();
    }

    virtual ofi_k_status_e get_itile_buf_size(const descriptor::kernel_arq_t &r_arg,
                                              descriptor::size_nds_t &r_itile_desc)
    {
        r_itile_desc[0].set2d(r_arg.input[0].tile_size[W], r_arg.input[0].tile_size[H]);
        r_itile_desc[1].set2d(r_arg.input[1].tile_size[W], r_arg.input[1].tile_size[H]);
        r_itile_desc[2].set2d(r_arg.input[2].tile_size[W], r_arg.input[2].tile_size[H] * 4);

        return OFI_SUCCESS;
    }

    virtual ofi_k_status_e get_otile_size(const descriptor::kernel_arq_t &r_arg, descriptor::size_descs_t &r_otile_desc)
    {
        int in_W = r_arg.input[2].size[W];
        int in_H = r_arg.input[2].size[H];

        int in_tile_W = 0;
        int in_tile_H = 0;
        int out_tile_W = 0;
        int out_tile_H = 0;
        int in_buf_size = 0;
        int out_buf_size = 0;

        if (in_W == 64 && in_H == 64) {
            in_tile_W = r_arg.input[2].tile_size[W]; // Grid W
            in_tile_H = r_arg.input[2].tile_size[H]; // Grid H
            out_tile_W = in_tile_W;
            out_tile_H = in_tile_H; // Grid H
        } else if (in_W == 64 && in_H == 32) {
            in_tile_W = r_arg.input[2].tile_size[W]; // Grid W
            in_tile_H = r_arg.input[2].tile_size[H]; // Grid H
            out_tile_W = in_tile_W;
            out_tile_H = 2 * in_tile_H; // Grid H
        } else if (in_W == 64 && in_H == 16) {
            in_tile_W = r_arg.input[2].tile_size[W]; // Grid W
            in_tile_H = r_arg.input[2].tile_size[H]; // Grid H
            out_tile_W = in_tile_W;
            out_tile_H = 4 * in_tile_H; // Grid H
        } else if (in_W == 32 && in_H == 64) {
            in_tile_W = r_arg.input[2].tile_size[W]; // Grid W
            in_tile_H = r_arg.input[2].tile_size[H]; // Grid H
            out_tile_W = 2 * in_tile_W;
            out_tile_H = in_tile_H; // Grid H
        } else if (in_W == 32 && in_H == 32) {
            in_tile_W = r_arg.input[2].tile_size[W]; // Grid W
            in_tile_H = r_arg.input[2].tile_size[H]; // Grid H
            out_tile_W = 2 * in_tile_W;
            out_tile_H = 2 * in_tile_H; // Grid H
        } else if (in_W == 32 && in_H == 16) {
            in_tile_W = r_arg.input[2].tile_size[W]; // Grid W
            in_tile_H = r_arg.input[2].tile_size[H]; // Grid H
            out_tile_W = 2 * in_tile_W;
            out_tile_H = 4 * in_tile_H; // Grid H
        } else if (in_W == 16 && in_H == 16) {
            in_tile_W = r_arg.input[2].tile_size[W]; // Grid W
            in_tile_H = r_arg.input[2].tile_size[H]; // Grid H
            out_tile_W = 4 * in_tile_W;
            out_tile_H = 4 * in_tile_H; // Grid H
        }

        in_buf_size = 4 * in_tile_H;
        out_buf_size = 4 * out_tile_H;

        //printf("TTT get_otile_size - in_W : %d, in_H : %d \n", in_W, in_H);
        //printf("TTT get_otile_size - in_tile_W : %d, in_tile_H : %d \n", in_tile_W, in_tile_H);
        //printf("TTT get_otile_size - out_tile_W : %d, out_tile_H : %d \n", out_tile_W, out_tile_H);

        r_otile_desc[0].data_size.set2d(out_tile_W, out_tile_H);
        r_otile_desc[1].data_size.set2d(out_tile_W, out_tile_H);
        r_otile_desc[2].data_size.set2d(out_tile_W, out_tile_H);
        r_otile_desc[3].data_size.set2d(out_tile_W, out_tile_H);
        r_otile_desc[4].data_size.set2d(out_tile_W, out_tile_H * 12);
        r_otile_desc[5].data_size.set2d(out_tile_W, out_tile_H * 6);
        r_otile_desc[6].data_size.set2d(out_tile_W, out_tile_H * 3);
        r_otile_desc[7].data_size.set2d(NUM_OUTPUT_VALUES, 1);
        r_otile_desc[8].data_size.set2d(in_tile_W, in_tile_H);

        /* align set */
        r_otile_desc[0].buf_size.set2d(out_tile_W, out_buf_size);
        r_otile_desc[1].buf_size.set2d(out_tile_W, out_buf_size);
        r_otile_desc[2].buf_size.set2d(out_tile_W, out_buf_size);
        r_otile_desc[3].buf_size.set2d(out_tile_W, out_buf_size);
        r_otile_desc[4].buf_size.set2d(out_tile_W, out_buf_size * 12);
        r_otile_desc[5].buf_size.set2d(out_tile_W, out_buf_size * 6);
        r_otile_desc[6].buf_size.set2d(out_tile_W, out_buf_size * 3);
        r_otile_desc[7].buf_size.set2d(NUM_OUTPUT_VALUES, 4);
        r_otile_desc[8].buf_size.set2d(in_tile_W, in_buf_size);

        return OFI_SUCCESS;
    }

    virtual ofi_k_status_e get_tile_vm_shape(const descriptor::kernel_arq_t &r_arg,
                                             descriptor::tile_vm_shape_descs_t &r_shape_desc)
    {
        OFI_SET_SIZE2D(&r_shape_desc.itile[0], IN_TILE_W, 0);
        OFI_SET_SIZE2D(&r_shape_desc.itile[1], IN_TILE_W, 0);
        OFI_SET_SIZE2D(&r_shape_desc.itile[2], IN_TILE_W, 0);

        OFI_SET_SIZE2D(&r_shape_desc.otile[0], OUT_TILE_W, 0);
        OFI_SET_SIZE2D(&r_shape_desc.otile[1], OUT_TILE_W, 0);
        OFI_SET_SIZE2D(&r_shape_desc.otile[2], OUT_TILE_W, 0);
        OFI_SET_SIZE2D(&r_shape_desc.otile[3], OUT_TILE_W, 0);
        OFI_SET_SIZE2D(&r_shape_desc.otile[4], OUT_TILE_W, 0);
        OFI_SET_SIZE2D(&r_shape_desc.otile[5], OUT_TILE_W, 0);
        OFI_SET_SIZE2D(&r_shape_desc.otile[6], OUT_TILE_W, 0);
        OFI_SET_SIZE2D(&r_shape_desc.otile[7], NUM_OUTPUT_VALUES, 0);
        OFI_SET_SIZE2D(&r_shape_desc.otile[8], IN_TILE_W, 0);

        return OFI_SUCCESS;
    }

    virtual ofi_k_status_e get_output_size(const descriptor::kernel_arq_t &r_arg, descriptor::size_nds_t &r_output_desc)
    {
        r_output_desc[0].set2d(OUT_BUF_W, OUT_BUF_H);
        r_output_desc[1].set2d(OUT_BUF_W, OUT_BUF_H);
        r_output_desc[2].set2d(OUT_BUF_W, OUT_BUF_H);
        r_output_desc[3].set2d(OUT_BUF_W, OUT_BUF_H);
        r_output_desc[4].set2d(OUT_BUF_W, OUT_BUF_H * 12);
        r_output_desc[5].set2d(OUT_BUF_W, OUT_BUF_H * 6);
        r_output_desc[6].set2d(OUT_BUF_W, OUT_BUF_H * 3);
        r_output_desc[7].set2d(NUM_OUTPUT_VALUES, OUT_VALUE_H);
        r_output_desc[8].set2d(OUT_FSIZE_W, OUT_FSIZE_H);

        return OFI_SUCCESS;
    }

    virtual ofi_k_status_e get_input_padding(const descriptor::kernel_arq_t &args,
                                             descriptor::padding_descs_t &r_padding_desc)
    {
        r_padding_desc[0].mode = OFI_PAD_DISABLE;
        //OFI_SET_PADDING(&r_padding_desc[0].size, 0, 0, 0, 0);
        return OFI_SUCCESS;
    }

  protected:
  private:
};

#endif /* __THUMBNAIL_DESC_H__ */

