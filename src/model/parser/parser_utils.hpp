#ifndef SRC_MODEL_PARSER_PARSER_UTIL_HPP_
#define SRC_MODEL_PARSER_PARSER_UTIL_HPP_

#include <cstring>
#include <string>
#include <vector>
#include "common/enn_debug.h"
#include "common/enn_utils.h"
#include "common/enn_utils_buffer.hpp"
#include "model/parser/cgo/cgo_utils.hpp"
#include "model/schema/schema_cgo.h"
#include "model/raw/model.hpp"

namespace enn {
namespace model {

namespace NNC {
constexpr char TAG[] = "ENNC";

struct FileHeader {
    uint32_t reserved;
    char file_identifier[4];
};
};  // namespace NNC

namespace CGO {
constexpr uint32_t FB_MAGIC_ID = 0x0FF1100F;
constexpr uint32_t FB_MAGIC_ID_EXT = 0x0FF1110F;
constexpr uint32_t FB_MAGIC_ID_ELF = 0x0E0F0E0F;
constexpr uint32_t FB_MAGIC_ID_BIN = 0x7E7F7E7F;
constexpr uint32_t FB_MAGIC_ID_UCGO = 0xFACE4885;

struct FileHeader {
    uint32_t magic;
    uint32_t fbs_length;
    uint32_t dsp_length;
};
};  // namespace CGO

class ParserUtils {
public:
    template <typename HeaderType>
    static void get_model_header(const void* mem, HeaderType* header) {
        if (mem != nullptr) {
            memcpy(reinterpret_cast<void*>(header), mem, sizeof(HeaderType));
        }
    }

    static bool is_nnc_model(const void* mem) {
        NNC::FileHeader header;
        get_model_header(mem, &header);
        return (memcmp(NNC::TAG, header.file_identifier, 4) == 0);
    }

    static bool is_cgo_model(uint32_t magic) {
        switch (magic) {
            case CGO::FB_MAGIC_ID:
                ENN_DBG_COUT << "Type: CGO original type" << std::endl;
                return true;
            case CGO::FB_MAGIC_ID_EXT:
                ENN_DBG_COUT << "Type: CGO Extension type" << std::endl;
                return false;
            case CGO::FB_MAGIC_ID_UCGO:
                ENN_DBG_COUT << "Type: UCGO type" << std::endl;
                return false;
        }
        ENN_INFO_COUT << "Couldn't identify type (maybe not cgo)" << std::endl;
        return false;
    }

    static bool is_cgo_model(const void* mem) {
        CGO::FileHeader header;
        get_model_header(mem, &header);
        return is_cgo_model(header.magic);
    }

    static std::pair<int, int> identify_model(enn::util::BufferReader::UPtrType & modelbuf, ModelType* out_model_type) {
        constexpr int NUL_POS = -1;

        uint32_t file_size;
        auto loaded_buffer = std::make_unique<char []>(modelbuf->get_size());
        enn::model::CGO::FileHeader header;

        auto exit_func = [&](std::string msg, ModelType type, int start, int end) {
            ENN_INFO_COUT << msg << std::endl;
            *out_model_type = type;
            return std::make_pair(start, end);
        };

        if (modelbuf->copy_buffer(loaded_buffer.get()))
            return exit_func("File open Error", ModelType::NONE, NUL_POS, NUL_POS);

        file_size = modelbuf->get_size();
        ENN_DBG_PRINT("Model addr: %p, size: %d\n", loaded_buffer.get(), file_size);

        if (static_cast<size_t>(file_size) <= sizeof(header)) {
            return exit_func("File is too small", ModelType::NONE, NUL_POS, NUL_POS);
        }

        if (is_nnc_model(loaded_buffer.get())) {
            return exit_func("NNC identified", ModelType::NNC, 0, file_size);
        }

        get_model_header(loaded_buffer.get(), &header);

        if (is_cgo_model(header.magic)) {
            return exit_func("CGO identified", ModelType::CGO, sizeof(header), sizeof(header) + header.fbs_length);
        }

        return exit_func("Identify model failed", ModelType::NONE, 0, file_size);
    }
};

};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_PARSER_PARSER_UTIL_HPP_