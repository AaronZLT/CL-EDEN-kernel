#ifndef SRC_MODEL_PARSER_STRATEGY_NNC_PARSE_STRATEGY_HPP_
#define SRC_MODEL_PARSER_STRATEGY_NNC_PARSE_STRATEGY_HPP_

#include "model/schema/schema_nnc.h"
#include "model/schema/flatbuffers/flexbuffers.h"
#include "model/parser/strategy/strategy.hpp"
#include "model/raw/model.hpp"

namespace enn {
namespace model {

class NncParseStrategy : public ParseStrategy {
public:
    explicit NncParseStrategy(const std::shared_ptr<ModelMemInfo> model);

    void parse_operators() override;
    void parse_tensors() override;
    void parse_operator_options() override;
    void parse_attribute() override;
    void parse_control_option() override;
    void parse_graph_infos() override;
    void post_execute() override;
    std::shared_ptr<raw::Model> result() override;
    void print() override;

private:
    const TFlite::Model* tflite_model;
    raw::ModelBuilder model_builder = raw::ModelBuilder::build();

    std::unordered_map<uint32_t, uint32_t> tensor_to_binary_indexes;
    std::unordered_map<uint32_t, uint32_t> tensor_to_param_indexes;

    uint32_t graph_version = 0;
    bool use_shared_mem = false;
    bool use_legacy_adjacent_adaptor = true;

    inline void parse_npu_options(const flatbuffers::Vector<uint8_t>* data);
    inline void parse_dsp_options(const flatbuffers::Vector<uint8_t>* data);
    inline void parse_unified_options(const TFlite::ENN_UNIFIED_DEVICEOptions* options);

    inline void rearrange_inputs_for_operator(int32_t op_index, std::vector<int32_t>& inputs,
                                              std::vector<int32_t>& result_inputs, std::vector<int32_t>& result_binaries);
    inline void rearrange_outputs_for_operator(int32_t op_index, std::vector<int32_t>& outputs,
                                               std::vector<int32_t>& result_outputs, std::vector<int32_t>& result_binaries);

    // TODO: Disable in release version
    bool validate_operators();
    bool validate_tensors();
    bool validate_operator_options();
    bool validate_attribute();
    bool validate_control_option();
    bool validate_graph_infos();
};

};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_PARSER_STRATEGY_NNC_PARSE_STRATEGY_HPP_
