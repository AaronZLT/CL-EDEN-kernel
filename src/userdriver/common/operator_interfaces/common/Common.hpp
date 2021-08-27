#pragma once

#include <unordered_map>
#include "userdriver/common/operator_interfaces/common/Includes.hpp"

#if defined(__ANDROID__)
typedef __fp16 _Float16_t;
#elif (__linux__)
typedef float _Float16_t;
#endif

typedef void* DataPtr;

constexpr int32_t UNDEFINED = -1;

// a container for n-D Tensor's dimensions
typedef std::vector<uint32_t> NDims;

enum class DataOrder {
    NCHW = 0,
    NHWC = 1,
    OTHER = 2,
    SIZE
};

enum class DataOrderChangeType {
    NHWC2NCHW = 0,
    NCHW2NHWC = 1,
    NHWC2DHWC4 = 2,
    NCHW2DHWC4 = 3,
    DHWC42NHWC = 4,
    DHWC42NCHW = 5,
    OTHER = 6,
    SIZE
};

enum class DeviceType {
    CPU = 0,
    GPU = 1,
    DSP = 2,
    NPU = 3,
    SIZE
};

enum class PlatformType {
    AMD = 0,
    ARM = 1
};

enum class ArmArchType {
    BIFROST = 0,
    MAKALU = 1,
    VALHALL = 2
};

// ToDo(empire.jung): After integrating source file, change it to use in "schema_generated.h" or optimized.
enum class DataType {
    UNKNOWN = -1,
    FLOAT = 0,
    HALF = 1,
    INT32 = 2,
    INT8 = 3,
    UINT8 = 4,
    BOOL = 5,
    INT16 = 6,
    UINT16 = 7,
    FLOAT16 = 8,
    SIZE
};
enum class ColorType {
    RESET = 0,
    BLACK = 1,   /* Black */
    RED = 2,     /* Red */
    GREEN = 3,   /* Green */
    YELLOW = 4,  /* Yellow */
    BLUE = 5,    /* Blue */
    MAGENTA = 6, /* Magenta */
    CYAN = 7,    /* Cyan */
    WHITE = 8,   /* White */
    BOLDBLACK = 9,        /* Bold Black */
    BOLDRED = 10,          /* Bold Red */
    BOLDGREEN = 11,        /* Bold Green */
    BOLDYELLOW = 12,       /* Bold Yellow */
    BOLDBLUE = 13,         /* Bold Blue */
    BOLDMAGENTA = 14,      /* Bold Magenta */
    BOLDCYAN = 15,         /* Bold Cyan */
    BOLDWHITE = 16,        /* Bold White */
    SIZE
};
enum class PrecisionType {
    FP32 = 0,
    FP16 = 1,
    INT32 = 2,
    INT16 = 3,
    INT8 = 4,
    UINT8 = 5,
    SIZE
};

enum class PrecisionChangeMode {
    FP32_TO_FP16 = 0,
    FP16_TO_FP32 = 1,
    OTHER = 2,
    SIZE
};

enum class ComputeType {
    TFLite = 0,
    Caffe,
    SIZE
};

enum class Status {
    SUCCESS = 0,
    FAILURE = 1,
    CL_FAILURE = 2,
    INVALID_PARAMS = 3,
    SIZE
};

enum class BufferType {
    DEDICATED = 0,           // const tensor of operator
    INTER_SHARED_REUSE = 1,  // output of operator
    INTER_SHARED_NEW = 2,    // input of operator
    INTRA_SHARED = 3,        // temprary medium tensor used in operator
    SIZE
};

enum class StorageType {
    BUFFER = 0,
    TEXTURE = 1,
    SIZE
};

enum class Reducer { SUM, MIN, MAX, PROD, ALL, ANY, SIZE };

enum { N_NCHW = 0, C_NCHW = 1, H_NCHW = 2, W_NCHW = 3 };

enum { N_NHWC = 0, H_NHWC = 1, W_NHWC = 2, C_NHWC = 3 };

typedef struct {
    uint32_t image_height;
    uint32_t image_width;
    size_t bytes;
    PrecisionType precision;
} TextureDescriptor;

typedef struct QuantInfo {
    float scale;
    int32_t zeroPoint;
} QuantInfo;

typedef struct Dim4 {
    uint32_t n, c, h, w;
} Dim4;

typedef struct Dim2 {
    uint32_t h, w;
} Dim2;

typedef struct Pad4 {
    uint32_t t, r, b, l;
} Pad4;

typedef struct Pad8 {
    uint32_t n_t, n_b, t, r, b, l, c_t, c_b;
} Pad8;
//  n_t the padding top range  of batch; n_b the padding below range of batch;
//  t,r,b,l are the same as Pad4;
//  c_t the padding top range of channel; c_b the padding below range of channel

inline uint32_t getDim(const Dim4& dim4, const uint32_t& axis) {
    switch (axis) {
        case 0:
            return dim4.n;
        case 1:
            return dim4.c;
        case 2:
            return dim4.h;
        case 3:
            return dim4.w;
        default:
            return 0;
    }
}

inline Dim4 convertDimToNCHW(const Dim4& dim_nhwc) {
    Dim4 dim_nchw = {dim_nhwc.n, dim_nhwc.w, dim_nhwc.c, dim_nhwc.h};
    return dim_nchw;
}

inline Dim4 convertDimToNHWC(const Dim4& dim_nchw) {
    Dim4 dim_nhwc = {dim_nchw.n, dim_nchw.h, dim_nchw.w, dim_nchw.c};
    return dim_nhwc;
}

inline uint32_t getTensorSizeFromDims(const Dim4& dims) {
    return dims.n * dims.c * dims.h * dims.w;
}

inline uint32_t getTensorSizeFromDims(const NDims& ndim) {
    return std::accumulate(ndim.begin(), ndim.end(), \
                            1, std::multiplies<uint32_t>());
}

inline Dim4 extendToDim4(const std::vector<uint32_t>& from) {
    Dim4 to = {1, 1, 1, 1};
    if (from.empty()) {
        to = {0, 0, 0, 0};
        return to;
    }
    for (uint32_t i = 0; i < from.size(); ++i) {
        switch (i) {
        case 0: {to.n = from.at(i); break;}
        case 1: {to.c = from.at(i); break;}
        case 2: {to.h = from.at(i); break;}
        case 3: {to.w = from.at(i); break;}
        default: break;
        }
    }
    return to;
}

inline size_t alignTo(size_t src, size_t alignment) {
    return (src + alignment - 1) / alignment * alignment;
}

// Nice to have, ToDo(empire.jung, TBD): Check 'precision == PrecisionType::xxxx' again later.
inline size_t getTypeBytes(const DataType& data_type, const PrecisionType& precision) {
    switch (data_type) {
        case DataType::FLOAT: {
            if (precision == PrecisionType::FP16) {
                return sizeof(float) / 2;
            } else {
                return sizeof(float);
            }
        }
        case DataType::HALF:
        case DataType::FLOAT16: {
            if (precision == PrecisionType::FP32) {
                return sizeof(float);
            } else {
                return sizeof(float) / 2;
            }
        }
        case DataType::INT32:
            return sizeof(int32_t);
        case DataType::INT8:
            return sizeof(int8_t);
        case DataType::UINT8:
            return sizeof(uint8_t);
        case DataType::BOOL:
            return sizeof(bool);
        case DataType::UINT16:
            return sizeof(uint16_t);
        case DataType::INT16:
            return sizeof(int16_t);
        default:
            return 0;
    }
}

inline int findMaxFactor(int parent, int threshold) {
    if (parent == 1 || threshold == 1) {
        return 1;
    }
    int step = 1 + (parent & 1);
    if (step == 2) {
        threshold = (((threshold - 1) >> 1) << 1) + 1;
    }
    int i = threshold < parent ? threshold : parent;
    while (i <= threshold && i > 0) {
        if (parent % i == 0) {
            return i;
        }
        i = i - step;
    }
    return 1;
}

template <typename T>
std::shared_ptr<T> make_shared_array(size_t size) {
    return std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
}

// Nice to have, ToDo(empire.jung ,TBD): Optimized with operator() ==
inline bool isDimsSame(const Dim4 &dim_1, const Dim4 &dim_2) {
    if (dim_1.n == dim_2.n && dim_1.c == dim_2.c && dim_1.h == dim_2.h && dim_1.w == dim_2.w) {
        return true;
    }
    return false;
}

inline bool isDimsSame(const NDims &dim_1, const NDims &dim_2) {
    return dim_1 == dim_2;
}

inline uint32_t viewNDimsAt(const NDims& nd, const uint32_t idx) {
    return idx < nd.size() ? nd.at(idx) : 0;
}

// reorder 1-D, 2-D, and 3-D tensor's dims in a NHWC->NCHW way for broadcasting. e.g.
// 1-D:  [D] -> [1, D, 1, 1]
// 2-D:  [C, D] -> [1, D, 1, C]
// 3-D:  [B, C, D] -> [1, D, B, C]
inline bool reorder123DimsTo4DimsForBroadcast(const NDims& from, NDims& to) {
    if (from.size() < 4) {
        if (from.size() == 1)
            to = {1, from[0], 1, 1};
        else if (from.size() == 2)
            to = {1, from[1], 1, from[0]};
        else
            to = {1, from[2], from[0], from[1]};
        return true;
    }
    return false;
}

inline bool getBroadcastedDims(const NDims &indims_lhs, const NDims &indims_rhs, NDims &outdims) {
    if (indims_lhs == indims_rhs) {
        outdims = indims_lhs;
        return true;
    }
    uint32_t lhs_idx = 0, rhs_idx = 0;
    if (indims_lhs.size() > indims_rhs.size()) {
        outdims.insert(outdims.end(),
                       indims_lhs.begin(),
                       indims_lhs.begin() + indims_lhs.size() - indims_rhs.size());
        rhs_idx = 0;
        lhs_idx = indims_lhs.size() - indims_rhs.size();
    } else {
        outdims.insert(outdims.end(),
                       indims_rhs.begin(),
                       indims_rhs.begin() + indims_rhs.size() - indims_lhs.size());
        lhs_idx = 0;
        rhs_idx = indims_rhs.size() - indims_lhs.size();
    }
    for (; lhs_idx < indims_lhs.size(); ++lhs_idx, ++rhs_idx) {
        if (indims_lhs[lhs_idx] == indims_rhs[rhs_idx]) {
            outdims.push_back(indims_lhs[lhs_idx]);
        } else if (indims_lhs[lhs_idx] != indims_rhs[rhs_idx] && indims_rhs[rhs_idx] == 1) {
            outdims.push_back(indims_lhs[lhs_idx]);
        } else if (indims_lhs[lhs_idx] != indims_rhs[rhs_idx] && indims_lhs[lhs_idx] == 1) {
            outdims.push_back(indims_rhs[rhs_idx]);
        } else {
            return false;
        }
    }
    return true;
}

inline bool isBroadcastable(const NDims& from, const NDims& to) {
    uint32_t in_idx = 0, target_idx = 0;
    if (from.size() > to.size())
        return false;
    if (from == to)
        return true;
    if (from.size() < to.size())
        target_idx = to.size() - from.size();
    for (; target_idx < to.size(); ++target_idx, ++in_idx) {
        if (to[target_idx] != from[in_idx] && from[in_idx] != 1)
            return false;
    }
    return true;
}

inline bool isValidAxis(const int32_t axis, const int32_t rank) {
    return (axis <= rank - 1 && axis + rank >= 0);
}

inline bool calculateBroadcastedShape(const Dim4 &indim_1, const Dim4 &indim_2, Dim4 &out_dim) {
#define CALCULATEOUTDIM(dim_1, dim_2, dim_3)                        \
    if (dim_1 != dim_2 && dim_1 != 1 && dim_2 != 1) {               \
          return false;                                             \
    } else {                                                        \
          dim_3 = (dim_1 == 1)  ? dim_2 : dim_1;                    \
    }
    CALCULATEOUTDIM(indim_1.n, indim_2.n, out_dim.n)
    CALCULATEOUTDIM(indim_1.c, indim_2.c, out_dim.c)
    CALCULATEOUTDIM(indim_1.h, indim_2.h, out_dim.h)
    CALCULATEOUTDIM(indim_1.w, indim_2.w, out_dim.w)
    return true;
}

inline bool canDimsBroadcast(Dim4 &in_dim, const Dim4 &out_dim) {
#define BROADCAST(dim_1, dim_2)                 \
    if (dim_1 != dim_2) {                       \
        if (dim_1 == 1) {                       \
            dim_1 = dim_2;                      \
        } else {                                \
            return false;                       \
        }                                       \
    }
    BROADCAST(in_dim.n, out_dim.n);
    BROADCAST(in_dim.c, out_dim.c);
    BROADCAST(in_dim.h, out_dim.h);
    BROADCAST(in_dim.w, out_dim.w);
    return true;
}

inline int32_t positiveRemainder(int32_t dividend, int32_t divisor) {
    return (divisor + (dividend % divisor)) % divisor;
}

inline int32_t clampedIndex(int32_t index, int dim, bool pos_stride) {
    return pos_stride
           ? (index >= dim ? dim
                           : positiveRemainder(
                            std::min(std::max(index, -dim), dim), dim))
           : (index < -dim
              ? -1
              : positiveRemainder(
                            std::min(std::max(index, -dim), dim - 1), dim));
}

template <typename T, typename N>
T IntegralDivideRoundUp(T n, N divisor) {
    const T div = static_cast<T>(divisor);
    const T q = n / div;
    return n % div == 0 ? q : q + 1;
}

template <typename T, typename N>
T AlignByN(T number, N n) {
    return IntegralDivideRoundUp(number, n) * n;
}

inline int32_t GetBiggestDividerWithPriority(int32_t number, int max_divider) {
    if (number % 8 == 0 && 8 <= max_divider) {
        return 8;
    }
    if (number % 4 == 0 && 4 <= max_divider) {
        return 4;
    }
    if (number % 2 == 0 && 2 <= max_divider) {
        return 2;
    }
    for (int i = max_divider; i != 0; i--) {
        if (number % i == 0) {
            return i;
        }
    }
    return 1;
}
