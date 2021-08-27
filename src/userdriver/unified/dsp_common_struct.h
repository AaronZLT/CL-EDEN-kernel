//------------------------------------------------------------------------------
/// @file  dsp_common_struct.h
///
/// @brief  dsp v4 interface data struct
///
/// @section copyright_section Copyright
/// &copy; 2021, Samsung Electronics Co., Ltd.
//------------------------------------------------------------------------------

#include <stdint.h>
#include <map>
#include <string>

#ifndef USERDRIVER_DSP_DSP_PARSE_TABLE_H_
#define USERDRIVER_DSP_DSP_PARSE_TABLE_H_

constexpr int32_t VALUE_FD_INIT = -1;
constexpr uint32_t VALUE_IOVA_INIT = (uint32_t)VALUE_FD_INIT;

/* DspMemInfo.mappingIndex -1 is for Baaw,TSGD,Kernel params
 * which don't use mappingIndex */
constexpr uint32_t MAPPING_INDEX_INVALID = ((uint32_t)(-1));
/* If UCGO don't need communication with NCP, value is -1 for UCGO(NN) case. */
constexpr uint16_t UCGO_UID_INVALID = ((uint16_t)(-1));

/**
* @enum DspMemType_e
* @brief This enum is for Memory type
*/
enum DspMemType_e : uint16_t {
    DSP_GRAPH_BIN = 0,  /// Target specific graph data; core data for DSP firmware
    CMDQ_BIN,           /// Cmdq binary
    KERNEL_BIN_STR,         /// Kernel binary string
    INPUT,              /// Input
    OUTPUT,             /// Output
    TEMP,               /// Memory for dsp intermediate buffers
    WEIGHT,             /// Weight
    BIAS,               /// Bias
    SCALAR,             /// Scalar
    CUSTOM,             /// Custom type, meaning only by load type
    EMPTY = 0x7000,
};

/**
* @enum DspLoadType_e
* @brief This enum defines how memory is obtained
*/
enum DspLoadType_e : uint16_t {
    ALLOC = 0,          /// Memory allocated by User driver
    ALLOC_ZERO,         /// Memory allocated and init to zero
    ALLOC_LOAD,         /// Memory allocated by User driver and loaded from ucgo
    IMPORT,             /// Memory imported from enn
    BYPASS,             /// Do nothing for memory, bypass another infos
};

#pragma pack(push, 1)
/**
* @brief Dsp memory info
* @details Represents info about each memory
*/
struct DspMemInfo {
    enum DspMemType_e mem_type;     /// Memory type
    enum DspLoadType_e load_type;   /// Memory load type
    uint32_t index;                 /// Index of same memory type
    uint32_t size;                  /// Memory size
    uint32_t offset;                /// Memory offset
    uint32_t dataOffset;            /// Memory load offset on ucgo from end of DspMemInfo list
    int32_t mappingIndex;           /// Index mapping to CMDQ memory info, negative value means do not map
    uint8_t reserved[8];
};

/**
* @brief Dsp memory info header
* @details Includes variable size mem info list
*/
struct DspUcgoHeader {
    uint32_t magic;                 /// Magic id; BEEF9091
    uint32_t unique_id;             /// Ucgo unique id; to distinguish multi ucgo at one unified op
    uint32_t version[5];            /// Latest git hash
    uint32_t totalSize;             /// Total size of UCGO
    uint32_t numList;               /// Num of Mem info list
    uint8_t reserved[4];
    struct DspMemInfo info[0];      /// MemInfoList; variable size
};

typedef struct UnifiedDspCgo {
  uint32_t cgoMagic;  // 0x0FF1100F
  uint32_t sizeFlatbuffer;  // must be 0 for UnifiedDspCgo.
  uint32_t sizeTsgd;  // not used for UnifiedDspCgo.
  struct DspUcgoHeader header;
} UnifiedDspCgo_t;

#pragma pack(pop)

typedef enum ofi_mem_shape {
  OFI_SHAPE_WIDTH = 0,
  OFI_SHAPE_HEIGHT,
  OFI_SHAPE_CHANNEL,
  OFI_SHAPE_PIXELBIT,
} ofi_mem_shape_e;

// part of ofi_common_public.h
// geunwon.lee below structs must be replaced when vs4l for dsp is ready

typedef enum ofi_common_addr_type {
  OFI_V_ADDR,             ///< virtual address
  OFI_DV_ADDR,            ///< device virtual address (iova)
  OFI_FD,                 ///< File descriptor
} ofi_common_addr_type_e;

/**
 * @brief memory attributes
 *
 * @support : it support ofi_v1 / ofi_v2 / ofi_v3
 * @note    : OFI_UNKNWON_CACHEABLE support only ofi_v3
 */
typedef enum ofi_common_mem_attr {
  OFI_CACHEABLE,         ///< Allocate with cacheable
  OFI_NON_CACHEABLE,     ///< Allocate with non-cacheable
  OFI_UNKNOWN_CACHEABLE, ///< Allocate with unknown-cacheable
} ofi_common_mem_attr_e;

/**
 * @brief memory type
 *
 * @support : it support ofi_v1 / ofi_v2 / ofi_v3
 */
typedef enum ofi_common_mem_type {
  OFI_MEM_ION,            ///< ION allocator
  OFI_MEM_MALLOC,         ///< Malloc in x86
  OFI_MEM_ASHMEM,         ///< Ashmem in android
  OFI_MEM_NONE = 9,       ///< Not allocated
} ofi_common_mem_type_e;


#define NUM_ENN_ID (16)
#define NUM_UCGO_UID (16)

union global_id {
  struct {
    uint32_t ucgo_uid :  NUM_UCGO_UID; // LSB
    uint32_t enn_id : NUM_ENN_ID; // MSB
  } gid;
  uint32_t gid_num;
};

typedef struct ofi_v4_mem {
  uint8_t  addr_type;     ///< ofi_common_addr_type_e
  uint8_t  mem_attr;      ///< ofi_common_mem_attr_e
  uint8_t  mem_type;      ///< ofi_common_mem_type_e
  uint8_t  is_mandatory;  ///< is this buf should be allocated in prepare phase?
  uint32_t size;          ///< if size = 0, not allocated
  uint32_t offset;
  uint32_t reserved;
  uint32_t param[4];
  uint64_t vaddr;
  union {
    struct {
      int32_t  fd;        ///< file desciptor from ion/ashmem allocator
      uint32_t iova;      ///< device driver fill this field with fd
    } mem;
    void     *addr;       ///< used in x86
  } get_addr;
} ofi_v4_mem_t;

typedef struct ofi_v4_param {
  uint32_t      param_type;       ///< type of param
  union {                         ///< param detail
    uint32_t    param_index;      ///< index of param list for OFI_PARAM_UPDATE
    uint32_t    kernel_id;        ///< hash value for OFI_PARAM_KERNEL
  } idx;
  ofi_v4_mem_t  param_mem;         ///< param memory
} ofi_v4_param_t;

typedef struct ofi_v4_load_graph_info {
  uint32_t        global_id;       ///< global_id=(HEAD|TARGET|GRAPH_ID|-----)
  uint32_t        n_tsgd;          ///< num of tsgd_info
  uint32_t        n_param;         ///< num of param
  uint32_t        n_kernel;        ///< num of kernel binary
  ofi_v4_param_t  param_list[0];   ///< param list
} ofi_v4_load_graph_info_t;

typedef struct ofi_v4_execute_msg_info {
  uint32_t        global_id;        ///< global_id=(HEAD|TARGET|GRAPH_ID|MSG_ID)
  uint32_t        n_update_param;   ///< num of update_param
  ofi_v4_param_t  param_list[0];    ///< param list
} ofi_v4_execute_msg_info_t;


// part of dal_global.h
struct KernelBinInfo {
  uint32_t kernel_id;               ///< hash value of kernel binary string
  std::string kernel_path;          ///< Path of kernel binary
  KernelBinInfo(std::string kpath) {
    kernel_id = 0; // unused now
    kernel_path = kpath;
  }
  inline KernelBinInfo &operator=(const KernelBinInfo &info) {
    kernel_id = info.kernel_id;
    kernel_path = info.kernel_path;
    return *this;
  }
};
// end of dal_global.h

#endif
