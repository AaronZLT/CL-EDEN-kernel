#ifndef SRC_MODEL_RAW_DATA_NPU_OPTIONS_HPP_
#define SRC_MODEL_RAW_DATA_NPU_OPTIONS_HPP_

#include "common/enn_debug.h"
#include "model/raw/model.hpp"
#include "model/schema/flatbuffers/flexbuffers.h"

#define OPT_LEVEL(str) (str[0] == 'O' ? atoi(&str[1]) : -1)

namespace enn {
namespace model {
namespace raw {
namespace data {

class NPUOptions {
private:
    bool binding_ifm;              // false
    bool binding_ofm;              // false
    std::string compiled_command;  // ./nncontainer-gen -c NPU ...
    std::string framework;         // SCaffe
    bool hw_cfu;                   // false
    std::string name;              // npu binary name
    std::string model;             // Q_inception_v3_pruned48.protobin
    std::string ncp_version;       //
    std::string npuc_version;      // v1.6.6.i
    std::string onnx_name;         // MV2_Deeplab_V3_plus_MLPerf_tflite.onnx
    int32_t opt_level;             // O1
    std::string protobin;          // Q_inception_v3_pruned48.caffemodel
    std::string prototxt;          // Q_inception_v3_pruned48.prototxt
    std::string quant_bw;          // PAMIR
    std::string quant_dev;         //
    std::string soc_type;          // Pamir
    bool use_shared_mem;           // false

    class QuantizationMode {
    public:
        uint32_t bit_width_a;     // 8
        uint32_t bit_width_w;     // 8
        uint32_t bit_width_bias;  // 48
        uint32_t bit_width_nfu;   // 16
        uint32_t bit_width_c;     // 8
    } quant_mode;                 // bit_width_a=8 bit_width_w=8 bit_width_bias=48 bit_width_NFU=16 bit_width_c=8

    friend class NPUOptionsBuilder;

public:
    NPUOptions() : binding_ifm(false), binding_ofm(false), compiled_command(""), framework(""), hw_cfu(false), name("NPU"),
                   model(""), ncp_version(""), npuc_version(""), onnx_name(""), opt_level(UNDEFINED), protobin(""),
                   prototxt(""), quant_bw(""), quant_dev(""), soc_type(""), use_shared_mem(false),
                   quant_mode{0, 0, 0, 0, 0} {}

    bool is_binding_ifm() {
        return binding_ifm;
    }

    bool is_binding_ofm() {
        return binding_ofm;
    }

    std::string get_compiled_command() {
        return compiled_command;
    }

    std::string get_framework() {
        return framework;
    }

    bool is_use_hw_cfu() {
        return hw_cfu;
    }

    std::string get_name() {
        return name;
    }

    std::string get_model() {
        return model;
    }

    std::string get_ncp_version() {
        return ncp_version;
    }

    std::string get_npuc_version() {
        return npuc_version;
    }

    std::string get_onnx_name() {
        return onnx_name;
    }

    int32_t get_opt_level() {
        return opt_level;
    }

    std::string get_protobin() {
        return protobin;
    }

    std::string get_prototxt() {
        return prototxt;
    }

    std::string get_quant_bw() {
        return quant_bw;
    }

    std::string get_quant_dev() {
        return quant_dev;
    }

    QuantizationMode get_quant_mode() {
        return quant_mode;
    }

    std::string get_soc_type() {
        return soc_type;
    }

    bool is_use_shared_mem() {
        return use_shared_mem;
    }
};  // namespace data

class NPUOptionsBuilder : public ModelBuilder {
private:
    std::shared_ptr<NPUOptions> npu_options_;

    inline std::vector<std::string> name_split(std::string input, char delimiter) {
        std::vector<std::string> answer;
        std::stringstream ss(input);
        std::string temp;
        while (getline(ss, temp, delimiter)) {
            answer.push_back(temp);
        }
        return answer;
    }

    inline void parse_quant_mode(std::string value, NPUOptions::QuantizationMode& quant_mode) {
        std::vector<std::string> modes = name_split(value, ' ');
        if (modes.size() == 5) {
            for (size_t i = 0; i < modes.size(); ++i) {
                std::vector<std::string> mode = name_split(modes[i], '=');
                if (mode.size() == 2) {
                    if (i == 0)
                        quant_mode.bit_width_a = atoi(mode[1].c_str());
                    else if (i == 1)
                        quant_mode.bit_width_w = atoi(mode[1].c_str());
                    else if (i == 2)
                        quant_mode.bit_width_bias = atoi(mode[1].c_str());
                    else if (i == 3)
                        quant_mode.bit_width_nfu = atoi(mode[1].c_str());
                    else if (i == 4)
                        quant_mode.bit_width_c = atoi(mode[1].c_str());
                }
            }
        }
    }

public:
    explicit NPUOptionsBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};

    NPUOptionsBuilder& add_npu_options() {
        npu_options_ = std::make_unique<NPUOptions>();
        return *this;
    }

    NPUOptionsBuilder& get_npu_options(uint32_t index) {
        npu_options_ = raw_model_->get_npu_options().at(index);
        return *this;
    }

    NPUOptionsBuilder& set_npu_options(const flexbuffers::Map& m) {
        ENN_DBG_COUT << "npu_options.size() : " << m.size() << std::endl;

        for (size_t opt_idx = 0; opt_idx < m.size(); ++opt_idx) {
            auto key = m.Keys()[opt_idx].AsKey();
            auto val = m[key].AsString().c_str();

            ENN_DBG_PRINT("  - %s: %s\n", key, val);

            std::string key_string = std::string(key);
            if (key_string == "BINDING_IFM") {
                npu_options_->binding_ifm = strcmp(val, "false") ? true : false;
            } else if (key_string == "BINDING_OFM") {
                npu_options_->binding_ofm = strcmp(val, "false") ? true : false;
            } else if (key_string == "COMPILED_COMMAND") {
                npu_options_->compiled_command = val;
            } else if (key_string == "FRAMEWORK") {
                npu_options_->framework = val;
            } else if (key_string == "HW_CFU") {
                npu_options_->hw_cfu = strcmp(val, "false") ? true : false;
            } else if (key_string == "MODEL") {
                npu_options_->model = val;
            } else if (key_string == "NAME") {
                npu_options_->name = val;
            } else if (key_string == "NCP_VERSION") {
                npu_options_->ncp_version = val;
            } else if (key_string == "NPUC_VERSION") {
                npu_options_->npuc_version = val;
            } else if (key_string == "ONNX") {
                npu_options_->onnx_name = val;
            } else if (key_string == "OPT_LEVEL") {
                npu_options_->opt_level = OPT_LEVEL(val);
            } else if (key_string == "PROTOBIN") {
                npu_options_->protobin = val;
            } else if (key_string == "PROTOTXT") {
                npu_options_->prototxt = val;
            } else if (key_string == "QUANT_BW") {
                npu_options_->quant_bw = val;
            } else if (key_string == "QUANT_DEV") {
                npu_options_->quant_dev = val;
            } else if (key_string == "QUANT_MODE") {
                parse_quant_mode(val, npu_options_->quant_mode);
            } else if (key_string == "SOC_TYPE") {
                npu_options_->soc_type = val;
            } else {
                ENN_WARN_PRINT("  >>It's unknown metadata key : %s\n", key);
            }
        }

        return *this;
    }

    NPUOptionsBuilder& set_use_shared_mem(bool use_shared_mem) {
        npu_options_->use_shared_mem = use_shared_mem;
        return *this;
    }

    NPUOptionsBuilder& set_binding_ifm(bool is_binding_ifm) {
        npu_options_->binding_ifm = is_binding_ifm;
        return *this;
    }

    NPUOptionsBuilder& set_binding_ofm(bool is_binding_ofm) {
        npu_options_->binding_ofm = is_binding_ofm;
        return *this;
    }

    void build() {
        raw_model_->npu_options_.push_back(std::move(npu_options_));
    }
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_NPU_OPTIONS_HPP_
