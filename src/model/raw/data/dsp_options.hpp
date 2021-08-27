#ifndef SRC_MODEL_RAW_DATA_DSP_OPTIONS_HPP_
#define SRC_MODEL_RAW_DATA_DSP_OPTIONS_HPP_

#include "common/enn_debug.h"
#include "model/raw/model.hpp"
#include "model/schema/flatbuffers/flexbuffers.h"

namespace enn {
namespace model {
namespace raw {
namespace data {

class DSPOptions {
private:
    bool async_exec;   // false
    bool binding_ifm;  // false
    bool binding_ofm;  // false
    std::string name;  // dsp binary name

    friend class DSPOptionsBuilder;

public:
    DSPOptions() : async_exec(false), binding_ifm(false), binding_ofm(false), name("DSP") {}

    bool is_async_exec() {
        return async_exec;
    }

    bool is_binding_ifm() {
        return binding_ifm;
    }

    bool is_binding_ofm() {
        return binding_ofm;
    }

    std::string get_name() {
        return name;
    }

};  // namespace data

class DSPOptionsBuilder : public ModelBuilder {
private:
    std::shared_ptr<DSPOptions> dsp_options_;

    inline std::vector<std::string> name_split(std::string idspt, char delimiter) {
        std::vector<std::string> answer;
        std::stringstream ss(idspt);
        std::string temp;
        while (getline(ss, temp, delimiter)) {
            answer.push_back(temp);
        }
        return answer;
    }

public:
    explicit DSPOptionsBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};

    DSPOptionsBuilder& add_dsp_options() {
        dsp_options_ = std::make_unique<DSPOptions>();
        return *this;
    }

    DSPOptionsBuilder& get_dsp_options(uint32_t index) {
        dsp_options_ = raw_model_->get_dsp_options().at(index);
        return *this;
    }

    DSPOptionsBuilder& set_dsp_options(const flexbuffers::Map& m) {
        ENN_DBG_COUT << "dsp_options.size() : " << m.size() << std::endl;

        for (size_t opt_idx = 0; opt_idx < m.size(); ++opt_idx) {
            auto key = m.Keys()[opt_idx].AsKey();
            auto val = m[key].AsString().c_str();

            ENN_DBG_PRINT("  - %s: %s\n", key, val);

            std::string key_string = std::string(key);
            if (key_string == "ASYNC_EXEC") {
                dsp_options_->async_exec = strcmp(val, "false") ? true : false;
            } else if (key_string == "BINDING_IFM") {
                dsp_options_->binding_ifm = strcmp(val, "false") ? true : false;
            } else if (key_string == "BINDING_OFM") {
                dsp_options_->binding_ofm = strcmp(val, "false") ? true : false;
            } else if (key_string == "NAME") {
                dsp_options_->name = val;
            } else {
                ENN_WARN_PRINT("  >>It's unknown metadata key : %s\n", key);
            }
        }

        return *this;
    }

    void build() {
        raw_model_->dsp_options_.push_back(std::move(dsp_options_));
    }
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_DSP_OPTIONS_HPP_
