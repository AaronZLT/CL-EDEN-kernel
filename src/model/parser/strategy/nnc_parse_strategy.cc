#include <set>
#include <unordered_map>

#include "common/enn_debug.h"
#include "common/helper_templates.hpp"
#include "model/parser/strategy/nnc_parse_strategy.hpp"
#include "model/raw/data/attribute.hpp"
#include "model/raw/data/binary.hpp"
#include "model/raw/data/buffer.hpp"
#include "model/raw/data/graph_info.hpp"
#include "model/raw/data/model_option.hpp"
#include "model/raw/data/operator.hpp"
#include "model/raw/data/operator_options.hpp"
#include "model/raw/data/npu_options.hpp"
#include "model/raw/data/dsp_options.hpp"
#include "model/raw/data/tensor.hpp"

class TensorUtil {
public:
    static std::set<std::string> tflite_tensors;
    static std::set<std::string> binary_tensors;

    static bool is_NPU(std::string name) {
        return (name.compare("NCP") == 0 || name.compare("NPU") == 0 || name.compare("ENN_NPU") == 0);
    }

    static bool is_DSP(std::string name) {
        return (name.compare("DSP") == 0 || name.compare("ENN_DSP") == 0);
    }

    static bool is_UNIFIED(std::string name) {
        return (name.compare("ENN_UNIFIED_DEVICE") == 0);
    }

    static bool is_Shared_Mem(std::string name) {
        return (name.compare("Shared_Mem") == 0);
    }

    static bool is_tflite_tensor(std::string name) {
        return (tflite_tensors.find(name) != tflite_tensors.end());
    }

    static bool is_binary_tensor(std::string name) {
        return (binary_tensors.find(name) != binary_tensors.end());
    }

    static std::vector<std::string> name_split(std::string input, char delimiter) {
        std::vector<std::string> answer;
        std::stringstream ss(input);
        std::string temp;
        while (getline(ss, temp, delimiter)) {
            answer.push_back(temp);
        }
        return answer;
    }
};

std::set<std::string> TensorUtil::tflite_tensors = {
    "super:",       "FILTER",           "BIAS",                 "PADDING_SHAPE",    "PADDING_VALUE",        "LOOKUP",
    "KEY",          "VALUE",            "MEAN",                 "SCALE_OUT",        "ZERO_POINT_OUT",       "SCALE",
    "FRAC_LEN",     "ANCHORS",          "COLS_IN_CELL",         "LINES_IN_CELL",    "INTERLEAVED_SLICES",   "IDPS",
    "UNITSIZE",     "PERM",             "REDUCTION_INDICES",    "SPLIT_DIM",        "INDICES",              "UNPACK",
    "BEGIN",        "END",              "STRIDES",              "SIZE",             "NEW_SHAPE",            "AXIS",
    "Shared_Mem",   "CONSTANT_INPUT"
};

std::set<std::string> TensorUtil::binary_tensors = {
    "NCP",  // NPU binary name can be "NCP" or "NPU" by GraphGen
    "NPU",
    "DSP",
};

namespace enn {
namespace model {

NncParseStrategy::NncParseStrategy(const std::shared_ptr<ModelMemInfo> model) {
    ENN_DBG_PRINT("Model addr: %p, size: %d\n", model->va, model->size);

    TRY {
        tflite_model = TFlite::GetModel(model->va);

        flatbuffers::Verifier fbs_verifier(static_cast<uint8_t*>(const_cast<void*>(model->va)), model->size);

        verified = tflite_model->Verify(fbs_verifier);
        if (verified) {
            graph_version = tflite_model->version();
#ifdef SCHEMA_NNC_V1
            const uint32_t required_version = 3;  // It is the tflite schema version : Latest revision 3
            if (graph_version != required_version) {
                ENN_ERR_PRINT("TFlite schema version is wrong: correct(%d) != %d\n", required_version, graph_version);
                verified = false;
                return;
            }
#else
            uint32_t required_version = 200;  // It is the custom schema version (V2) for ENN : v2.0.0
            if (graph_version < required_version) {
                // ToDo(empire.jung, 7/30): Remove this temp code after merging new NNC files applyed with new version system
                required_version = 2;
                if (graph_version != required_version) {  // <-- Temporary condition for compatibility
                    ENN_ERR_PRINT("TFlite schema version is wrong: correct(%d) != %d\n", required_version, graph_version);
                    verified = false;
                    return;
                }                                         // <-- Temporary condition for compatibility
            }
#endif
            ENN_INFO_PRINT_FORCE("graph_version = %d\n", graph_version);
            model_mem_info = model;
        }
    }
    CATCH(what) {
        ENN_ERR_COUT << "TFlite model verify failed: " << what << std::endl;
        verified = false;
    }
    END_TRY
}

void NncParseStrategy::parse_operators() {
    auto tflite_subgraphs = tflite_model->subgraphs();            // SubGraph subGraphs[]
    auto tflite_subgraph = tflite_subgraphs->Get(0);              // SubGraph subgraph = subgraphs[subgraphIdx]
    auto tflite_operators = tflite_subgraph->operators();         // Operator operators[]
    auto tflite_operator_codes = tflite_model->operator_codes();  // OperatorCode operator_codes[]

    for (uint32_t operator_idx = 0; operator_idx < tflite_operators->size(); operator_idx++) {
        uint32_t op_index;
        std::string op_name;
        int32_t op_code = UNDEFINED;
        std::vector<int32_t> input_indexes;
        std::vector<int32_t> output_indexes;
        Accelerator accelerator = Accelerator::NONE;

        auto tflite_operator = tflite_operators->Get(operator_idx);
        uint32_t opcode_idx = tflite_operator->opcode_index();
        const char* operator_name = "";

        auto tflite_custom_operator = tflite_operator_codes->Get(opcode_idx)->custom_code();
        if (tflite_custom_operator) {
            operator_name = tflite_custom_operator->c_str();
        } else {
            auto tflite_builtin_operator = tflite_operator_codes->Get(opcode_idx)->builtin_code();
            operator_name = EnumNameBuiltinOperator(tflite_builtin_operator);
            op_code = static_cast<int32_t>(tflite_builtin_operator);
        }

#ifdef SCHEMA_NNC_V1
        if (TensorUtil::is_NPU(operator_name)) {
            accelerator = Accelerator::NPU;
        } else if (TensorUtil::is_DSP(operator_name)) {
            accelerator = Accelerator::DSP;
        }

        accelerator = (accelerator == Accelerator::NONE) ? Accelerator::CPU : accelerator;
#else
        accelerator = static_cast<Accelerator>(tflite_operator->target_hw());

        // (Temp) Set target_hw as CPU for DETECTION operator
        if (op_code == TFlite::BuiltinOperator_ENN_DETECTION) {
            accelerator = Accelerator::CPU;
        }
#endif

        op_index = operator_idx;
        op_name = std::string(operator_name);

        for (auto in : *tflite_operator->inputs()) {
            input_indexes.push_back(in);
        }

        for (auto out : *tflite_operator->outputs()) {
            output_indexes.push_back(out);
        }

        model_builder.build_operator()
            .add_operator()
            .set_op_index(op_index)
            .set_op_name(op_name)
            .set_op_code(op_code)
            .set_input_indexes(input_indexes)
            .set_output_indexes(output_indexes)
            .set_accelerator(accelerator)
            .build();
    }
}

bool NncParseStrategy::validate_operators() {
    // TODO: Implement
    return true;
}

void NncParseStrategy::parse_tensors() {
    std::string name = "";
    int32_t type = 0;
    uint32_t buffer_index = 0;
    int32_t binary_index = 0;
    Accelerator accelerator = Accelerator::NONE;

    std::unordered_map<std::string, std::tuple<const uint8_t*, uint32_t>> binary_data_holder;

    auto tflite_buffers = tflite_model->buffers();      // Buffer buffers[]
    auto tflite_subgraphs = tflite_model->subgraphs();  // SubGraph subGraphs[]
    auto tflite_subgraph = tflite_subgraphs->Get(0);    // SubGraph subgraph = subgraphs[subgraphIdx]
    auto tflite_tensors = tflite_subgraph->tensors();   // tensors:[Tensor]

    for (uint32_t tidx = 0; tidx < tflite_tensors->size(); tidx++) {
        auto tflTensor = tflite_tensors->Get(tidx);  // Tensor tensor = tensors[tidx]

        name = tflTensor->name()->c_str();
        type = tflTensor->type();
        buffer_index = tflTensor->buffer();

        // Building Binary (NCP/DSP)
        if (TensorUtil::is_binary_tensor(name.substr(0, 3))) {
            auto split = TensorUtil::name_split(name, '_');
            std::string target = split[0];    // "NPU,NCP" or "DSP"
            std::string category = split[1];  // "BINARY" or "NAME"
            auto tflite_buffer = tflite_buffers->Get(buffer_index);
            auto tflite_data = tflite_buffer->data();
            const uint8_t* data = tflite_data->data();
            int32_t size = tflite_data->size();

#ifdef SCHEMA_NNC_V1
            binary_index = std::distance(binary_data_holder.begin(), binary_data_holder.find(target));

            if (!util::is_map_contain(binary_data_holder, target)) {
                binary_data_holder[target] = std::make_tuple(data, size);
                tensor_to_binary_indexes[tidx] = binary_index;
            } else {
                char bin_name[128] = {0};
                const uint8_t* data_addr;
                int32_t data_size = 0;
                if (category == "BINARY") {
                    snprintf(bin_name, std::get<1>(binary_data_holder[category]),
                            "%s", std::get<0>(binary_data_holder[target]));
                    data_addr = data;
                    data_size = size;
                } else {
                    snprintf(bin_name, size, "%s", data);
                    data_addr = std::get<0>(binary_data_holder[target]);
                    data_size = std::get<1>(binary_data_holder[target]);
                }

                if (TensorUtil::is_NPU(target)) {
                    accelerator = Accelerator::NPU;
                } else if (TensorUtil::is_DSP(target)) {
                    accelerator = Accelerator::DSP;
                }

                model_builder.build_binary()
                    .add_binary()
                    .set_index(binary_index)
                    .set_name(bin_name)
                    .set_address(reinterpret_cast<uint8_t*>(model_mem_info->va))
                    .set_fd(model_mem_info->fd)
                    .set_offset(data_addr - reinterpret_cast<uint8_t*>(model_mem_info->va))
                    .set_size(data_size)
                    .set_buffer_index(buffer_index)
                    .set_accelerator(accelerator)
                    .build();

                tensor_to_binary_indexes[tidx] = binary_index;
            }
#else
            if (category == "BINARY") {
                binary_data_holder[target] = std::make_tuple(data, size);

                if (TensorUtil::is_NPU(target)) {
                    accelerator = Accelerator::NPU;
                } else if (TensorUtil::is_DSP(target)) {
                    accelerator = Accelerator::DSP;
                }

                model_builder.build_binary()
                    .add_binary()
                    .set_index(binary_index)
                    .set_name(target)
                    .set_address(reinterpret_cast<uint8_t*>(model_mem_info->va))
                    .set_fd(model_mem_info->fd)
                    .set_offset(data - reinterpret_cast<uint8_t*>(model_mem_info->va))
                    .set_size(size)
                    .set_buffer_index(buffer_index)
                    .set_accelerator(accelerator)
                    .build();

                tensor_to_binary_indexes[tidx] = binary_index++;
            }
#endif
            continue;
        }

        int32_t prev_operator_index = UNDEFINED;
        std::vector<int32_t> next_operator_indexes;

// #ifndef SCHEMA_NNC_V1
//         if (tflTensor->next_operators() != nullptr) {
//             use_legacy_adjacent_adaptor = false;
//             prev_operator_index = tflTensor->pre_operator();
//             next_operator_indexes = util::convert_vector<int32_t>(tflTensor->next_operators());
//         }
// #endif

        // To check using shared memory between NPU and DSP
        use_shared_mem = TensorUtil::is_Shared_Mem(name);

        // Building Tensor (like as Featurmap and Parameter)
        int32_t buffer_size = pixel_bit_format_size[static_cast<TFlite::TensorType>(type)];
        std::vector<uint32_t> shape = util::convert_vector<uint32_t>(tflTensor->shape());
        for (uint32_t size : shape) {
            buffer_size *= size;
        }

        const uint8_t* buffer_data = nullptr;
        if (tflite_buffers->size() > 0) {
            auto tflite_buffer = tflite_buffers->Get(buffer_index);
            auto tflite_data = tflite_buffer->data();
            buffer_data = tflite_data->data();
        } else {
            ENN_WARN_PRINT("Tensor(%s) has no buffer\n", name.c_str());
        }

        model_builder.build_tensor()
            .add_tensor()
            .set_index(tidx)
            .set_name(name)
            .set_type(type)
            .set_prev_operator_index(prev_operator_index)
            .set_next_operator_indexes(next_operator_indexes)
            .set_shape(shape)
            .set_quantization_parameters(tflTensor->quantization())
#ifndef SCHEMA_NNC_V1
            .set_symm_per_channel_quant_parameters(tflTensor->extram_param())
#endif
            .set_address(buffer_data)
            .set_size(buffer_size)
            .build();
    }
}

bool NncParseStrategy::validate_tensors() {
    // TODO: Implement
    return true;
}

inline void NncParseStrategy::parse_npu_options(const flatbuffers::Vector<uint8_t>* data) {
    if (data) {
        const flexbuffers::Map& m = flexbuffers::GetRoot(data->Data(), data->size()).AsMap();
        model_builder.build_npu_options().add_npu_options().set_npu_options(m).build();
    }
}

inline void NncParseStrategy::parse_dsp_options(const flatbuffers::Vector<uint8_t>* data) {
    if (data) {
        const flexbuffers::Map& m = flexbuffers::GetRoot(data->Data(), data->size()).AsMap();
        model_builder.build_dsp_options().add_dsp_options().set_dsp_options(m).build();
    }
}

inline void NncParseStrategy::parse_unified_options(const TFlite::ENN_UNIFIED_DEVICEOptions* options) {
    if (options) {
        for (auto option : *options->options()) {
            if (option->target_hw() == TFlite::TargetHw_NPU) {
                parse_npu_options(option->metadata());
            } else if (option->target_hw() == TFlite::TargetHw_DSP) {
                parse_dsp_options(option->metadata());
            }
        }
    }
}

void NncParseStrategy::parse_operator_options() {
    auto tflite_subgraphs = tflite_model->subgraphs();            // SubGraph subGraphs[]
    auto tflite_subgraph = tflite_subgraphs->Get(0);              // SubGraph subgraph = subgraphs[subgraphIdx]
    auto tflite_operators = tflite_subgraph->operators();         // Operator operators[]
    auto tflite_operator_codes = tflite_model->operator_codes();  // OperatorCode operator_codes[]

    for (uint32_t operator_idx = 0; operator_idx < tflite_operators->size(); operator_idx++) {
        auto tflite_operator = tflite_operators->Get(operator_idx);
        uint32_t opcode_idx = tflite_operator->opcode_index();
        std::string operator_name;

        auto tflite_custom_operator = tflite_operator_codes->Get(opcode_idx)->custom_code();
        if (tflite_custom_operator) {
            operator_name = tflite_custom_operator->c_str();
        } else {
            auto tflite_builtin_operator = tflite_operator_codes->Get(opcode_idx)->builtin_code();
            operator_name = EnumNameBuiltinOperator(tflite_builtin_operator);
        }

        auto builtin_options = tflite_operator->builtin_options_type();
        if (builtin_options != TFlite::BuiltinOptions_NONE) {
            uint32_t options_num = builtin_options;
            std::string name = EnumNameBuiltinOptions(builtin_options);
            const void* options = tflite_operator->builtin_options();

#ifdef SCHEMA_NNC_V1
            if (TensorUtil::is_NPU(operator_name)) {
                parse_npu_options(tflite_operator->custom_options());
            }
#else
            if (TensorUtil::is_NPU(operator_name)) {
                auto npu_options = static_cast<const TFlite::ENN_NPUOptions*>(tflite_operator->builtin_options());
                parse_npu_options(npu_options->metadata());
            } else if (TensorUtil::is_DSP(operator_name)) {
                auto dsp_options = static_cast<const TFlite::ENN_DSPOptions*>(tflite_operator->builtin_options());
                parse_dsp_options(dsp_options->metadata());
            } else if (TensorUtil::is_UNIFIED(operator_name)) {
                auto options = static_cast<const TFlite::ENN_UNIFIED_DEVICEOptions*>(tflite_operator->builtin_options());
                parse_unified_options(options);
            }
#endif

            if (options != nullptr) {
                model_builder.build_operator_options()
                    .add_operator_options()
                    .set_operator_index(operator_idx)
                    .set_num(options_num)
                    .set_name(name)
                    .set_options(options)
                    .build();
            }
        }
    }
}

bool NncParseStrategy::validate_operator_options() {
    // TODO: Implement
    return true;
}

void NncParseStrategy::parse_attribute() {
    int32_t model_type = static_cast<int32_t>(ModelType::NNC);

    model_builder.build_attribute()
        .add_attribute()
        .set_version(graph_version)
        .set_model_type(model_type)
        // Nice to have, ToDo(empire.jung, TBD): set value in future, if needed
        // .set_nn_api_type(nn_api_type)
        .build();

    bool relax_computation_float32_to_float16 = tflite_model->relaxComputationFloat32toFloat16();
    int32_t index = 0;
    int32_t legacy_model = tflite_model->compatible()->Get(index);
    model_builder.build_model_option()
        .add_model_option()
        .set_index(index)
        .set_legacy_model(legacy_model)
        .set_relax_computation_float32_to_float16(relax_computation_float32_to_float16)
        .build();
}

bool NncParseStrategy::validate_attribute() {
    // TODO: Implement
    return true;
}

void NncParseStrategy::parse_control_option() {
    // TODO: Implement
}

bool NncParseStrategy::validate_control_option() {
    // TODO: Implement
    return true;
}

void NncParseStrategy::parse_graph_infos() {
    auto tflite_subgraphs = tflite_model->subgraphs();

    for (uint32_t si = 0; si < tflite_subgraphs->size(); ++si) {
        auto tflite_subgraph = tflite_subgraphs->Get(si);

        std::string name = "NULL";
        if (tflite_subgraph->name() != nullptr) {
            name = tflite_subgraph->name()->c_str();
        }

        auto tflite_inputs = tflite_subgraph->inputs();
        std::vector<int32_t> inputs;
        for (uint32_t ii = 0; ii < tflite_inputs->size(); ++ii) {
            inputs.push_back(tflite_inputs->Get(ii));
        }

        auto tflite_outputs = tflite_subgraph->outputs();
        std::vector<int32_t> outputs;
        for (uint32_t oi = 0; oi < tflite_outputs->size(); ++oi) {
            outputs.push_back(tflite_outputs->Get(oi));
        }

        model_builder.build_graph_info().add_graph_info().set_name(name).set_inputs(inputs).set_outputs(outputs).build();
    }
}

bool NncParseStrategy::validate_graph_infos() {
    // TODO: Implement
    return true;
}

inline void NncParseStrategy::rearrange_inputs_for_operator(int32_t op_index, std::vector<int32_t>& inputs,
                                                            std::vector<int32_t>& result_inputs,
                                                            std::vector<int32_t>& result_binaries) {
    for (uint32_t idx : inputs) {
        if (util::is_map_contain(tensor_to_binary_indexes, idx)) {
            result_binaries.push_back(tensor_to_binary_indexes[idx]);
        } else {
            result_inputs.push_back(idx);

            if (use_legacy_adjacent_adaptor) {
                model_builder.build_tensor().get_tensor_from_index(idx).add_next_operator_index(op_index);
            }
        }
    }
}
inline void NncParseStrategy::rearrange_outputs_for_operator(int32_t op_index, std::vector<int32_t>& outputs,
                                                             std::vector<int32_t>& result_outputs,
                                                             std::vector<int32_t>& result_binaries) {
    for (uint32_t idx : outputs) {
        if (util::is_map_contain(tensor_to_binary_indexes, idx)) {
            result_binaries.push_back(tensor_to_binary_indexes[idx]);
        } else {
            result_outputs.push_back(idx);

            if (use_legacy_adjacent_adaptor) {
                model_builder.build_tensor().get_tensor_from_index(idx).set_prev_operator_index(op_index);
            }
        }
    }
}

void NncParseStrategy::post_execute() {
    auto parsed_model = model_builder.get_model();
    /*
     * Step1) Remove excepted index from operator ifm_indexes/ofm_indexes
     *        Add parameter indexes to each operator
     *        Add binary index to each operator
     */
    for (size_t op_idx = 0; op_idx < parsed_model->get_operators().size(); op_idx++) {
        auto operator_ = parsed_model->get_operators().at(op_idx);

        std::vector<int32_t> new_input_indexes;
        std::vector<int32_t> new_output_indexes;
        std::vector<int32_t> new_binary_indexes;

        rearrange_inputs_for_operator(op_idx, operator_->get_input_indexes(), new_input_indexes, new_binary_indexes);
        rearrange_outputs_for_operator(op_idx, operator_->get_output_indexes(), new_output_indexes, new_binary_indexes);

        model_builder.build_operator()
            .get_operator(op_idx)
            .set_input_indexes(new_input_indexes)
            .set_output_indexes(new_output_indexes)
            .set_binary_indexes(new_binary_indexes);
    }

    /*
     * Step2) Add operator options index to each operator
     */
    int options_index = 0;
    for (auto options : parsed_model->get_operator_options()) {
        int operator_index = options->get_operator_index();
        if (operator_index >= 0) {
            model_builder.build_operator().get_operator(operator_index).set_operator_options_index(options_index);
        }
        options_index++;
    }

    /*
     * Step3) Set each binary name from binary options
     */
    uint32_t npu_idx = 0, dsp_idx = 0;
    for (size_t bidx = 0; bidx < parsed_model->get_binaries().size(); ++bidx) {
        auto binary = parsed_model->get_binaries().at(bidx);
#ifndef SCHEMA_NNC_V1
        if (TensorUtil::is_NPU(binary->get_name()) && parsed_model->get_npu_options().size() > npu_idx) {
            std::string binary_name = parsed_model->get_npu_options().at(npu_idx)->get_name();
            model_builder.build_binary().get_binary(bidx).set_name(binary_name);
            model_builder.build_npu_options().get_npu_options(npu_idx).set_use_shared_mem(use_shared_mem);
            npu_idx++;
        } else if (TensorUtil::is_DSP(binary->get_name()) && parsed_model->get_dsp_options().size() > dsp_idx) {
            std::string binary_name = parsed_model->get_dsp_options().at(dsp_idx)->get_name();
            model_builder.build_binary().get_binary(bidx).set_name(binary_name);
            dsp_idx++;
        }
#endif
    }
}

void NncParseStrategy::print() {
    auto parsed_model = model_builder.get_model();

    for (size_t i = 0; i < parsed_model->get_graph_infos().size(); ++i) {
        auto graph_info = parsed_model->get_graph_infos().at(i);
        ENN_DBG_PRINT("GraphInfo[%zu] input_size: %zu, output_size: %zu\n", i, graph_info->get_inputs().size(),
                      graph_info->get_outputs().size());
    }

    auto model_option = parsed_model->get_model_options();
    ENN_DBG_PRINT("ModelOption[%d] legacy_model = %d, relax_computation_float32_to_float16 = %s\n",
                  model_option->get_index(), model_option->get_legacy_model(),
                  model_option->is_relax_computation_float32_to_float16() ? "true" : "false");

    for (size_t op_idx = 0; op_idx < parsed_model->get_operators().size(); op_idx++) {
        auto operator_ = parsed_model->get_operators().at(op_idx);
        ENN_DBG_PRINT("Operator[%2zu] (hw:%d) (code: %4d) %s\n", op_idx, (int)operator_->get_accelerator(),
                      operator_->get_op_code(), operator_->get_name().c_str());
    }

    for (auto options : parsed_model->get_operator_options()) {
        ENN_DBG_PRINT("builtin_options[%2d] (code: %4d) %s\n", options->get_operator_index(), options->get_number(),
                      options->get_name().c_str());
    }

    ENN_DBG_COUT << "(prev) -> [Tensor] -> (next)    Size    Name" << std::endl;
    ENN_DBG_COUT << "--------------------------------------------" << std::endl;
    for (auto tensor : parsed_model->get_tensors()) {
        std::string prev = "";
        if (tensor->get_prev_operator_index() != UNDEFINED) {
            prev = std::to_string(tensor->get_prev_operator_index()) + " ";
        }
        std::string next = "";
        for (auto next_op_index : tensor->get_next_operator_indexes()) {
            next += std::to_string(next_op_index) + " ";
        }
        ENN_DBG_PRINT("%6s -> [ %4d ] -> %6s%9d   %s\n", prev.c_str(), tensor->get_index(), next.c_str(), tensor->get_size(),
                      tensor->get_name().c_str());
    }

    for (size_t bidx = 0; bidx < parsed_model->get_binaries().size(); ++bidx) {
        auto binary = parsed_model->get_binaries().at(bidx);
        ENN_DBG_PRINT("Binary[%zu] idx: %d, name: %s, fd: %d, offset: %d, addr: %p, size: %d, accelerator: %d\n", bidx,
                      binary->get_index(), binary->get_name().c_str(), binary->get_fd(), binary->get_offset(),
                      binary->get_address(), binary->get_size(), static_cast<int>(binary->get_accelerator()));
    }

    ENN_DBG_PRINT("use_shared_mem : %s\n", use_shared_mem ? "True" : "False");
}

std::shared_ptr<raw::Model> NncParseStrategy::result() {
    return model_builder.create();
}

};  // namespace model
};  // namespace enn
