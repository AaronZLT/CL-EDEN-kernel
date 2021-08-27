#ifndef SRC_MODEL_PARSER_CGO_UTIL_HPP_
#define SRC_MODEL_PARSER_CGO_UTIL_HPP_

#include "common/enn_debug.h"
#include "model/schema/schema_cgo.h"
#include "model/raw/model.hpp"
#include "model/types.hpp"

namespace enn {
namespace model {

class CgoParameter {
public:
    CgoParameter() : index(UNDEFINED), name(""), offset(0), size(0), is_loaded_from_file(false) {}

    CgoParameter(int32_t index_, std::string name_, uint32_t offset_, uint32_t size_, bool is_loaded_from_file_) {
        index = index_;
        name = name_;
        offset = offset_;
        size = size_;
        is_loaded_from_file = is_loaded_from_file_;
    }

    int32_t index;
    std::string name;
    uint32_t offset;
    uint32_t size;
    bool is_loaded_from_file;
};

using CgoParameterList = std::vector<CgoParameter>;

class CgoUtils {
public:
    static std::shared_ptr<CgoParameterList> parse_parameters(const void* va, const int size) {
        ENN_INFO_COUT << "va: " << va << ", size: " << size << std::endl;

        // get top of ofi raw graph
        auto raw_graph = ofi::rawgraph::Getfb_OfiRawGraph(va);

        // verification
        flatbuffers::Verifier fbs_verifier(static_cast<uint8_t*>(const_cast<void*>(va)), size);
        CHECK_AND_RETURN_ERR(!raw_graph->Verify(fbs_verifier), nullptr, "Input file is not verified(cgo)\n");

        // get dsp binary buffer
        auto ti_ = raw_graph->core()->target_info();
        if (ti_ == 0) {
            ENN_ERR_PRINT_FORCE("No target(DSP) binary information in model, please check\n");
            return nullptr;
        }
        if (ti_->type() != ofi::rawgraph::fb_OfiGraphType_OFI_GRAPH_TYPE_CVNN2019) {
            ENN_ERR_PRINT_FORCE("DSP binary type is old (not CVNN2019 type), please check\n");
            return nullptr;
        }

        std::shared_ptr<CgoParameterList> parsed_list = std::make_shared<CgoParameterList>();

        auto ti = ti_->dsp2019();
        auto dsp_buf_info = ti->graph_info();
        parsed_list->push_back(CgoParameter(0, std::string(dsp_buf_info->name()->c_str()), dsp_buf_info->buf_info(),
                                            dsp_buf_info->size(), true));

        // get parameters
        auto param_list = raw_graph->param()->param_list();
        for (int32_t idx = 0; idx < param_list->size(); ++idx) {
            auto param_ele = param_list->Get(idx);
            auto buf_info = param_ele->buf_info();
            if (buf_info->load_default() == ofi::rawgraph::fb_OfiMemLoadType_OFI_MEM_LOAD_FROM_CGO) {
                parsed_list->push_back(CgoParameter(idx, std::string(buf_info->name()->c_str()), buf_info->buf_info(),
                                                    buf_info->size(), true));
            } else if (param_ele->is_user_defined() && param_ele->is_scalar() && param_ele->is_allocate_every_session()) {
                parsed_list->push_back(CgoParameter(idx, std::string(buf_info->name()->c_str()), buf_info->buf_info(),
                                                    buf_info->size(), false));
            } else if (strncmp(buf_info->name()->c_str(), "TEMP", 4) == 0) {
                parsed_list->push_back(CgoParameter(idx, std::string(buf_info->name()->c_str()), buf_info->buf_info(),
                                                    buf_info->size(), false));
            } else {
                // size = 0 means, send no buffer to service
                parsed_list->push_back(CgoParameter(idx, std::string(buf_info->name()->c_str()), buf_info->buf_info(),
                                                    0, false));
            }
        }

        // print
        for (size_t i = 0; i < parsed_list->size(); ++i) {
            auto ele = parsed_list->at(i);
            ENN_INFO_PRINT("[%d] name(%20s), offset(%10d), size(%10d), %16s\n", ele.index, ele.name.c_str(), ele.offset,
                           ele.size, ele.is_loaded_from_file ? "LOADED_FROM_FILE" : "NOT LOAD");
        }

        return std::move(parsed_list);
    }
};

};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_PARSER_CGO_UTIL_HPP_