#include "common/enn_debug.h"
#include "common/helper_templates.hpp"
#include "model/parser/strategy/cgo_parse_strategy.hpp"
#include "model/parser/cgo/dsp_kernel_table.hpp"
#include "model/raw/data/attribute.hpp"
#include "model/raw/data/binary.hpp"
#include "model/raw/data/graph_info.hpp"
#include "model/raw/data/operator.hpp"
#include "model/raw/data/scalar.hpp"
#include "model/raw/data/tensor.hpp"

namespace enn {
namespace model {

CgoParseStrategy::CgoParseStrategy(const std::shared_ptr<ModelMemInfo> model,
                                   const std::shared_ptr<std::vector<std::shared_ptr<ModelMemInfo>>> params) {
    ENN_DBG_PRINT("Model addr: %p, size: %d\n", model->va, model->size);

    TRY {
        ofi_raw_graph = ofi::rawgraph::Getfb_OfiRawGraph(model->va);

        flatbuffers::Verifier fbs_verifier(static_cast<uint8_t*>(const_cast<void*>(model->va)), model->size);

        verified = ofi_raw_graph->Verify(fbs_verifier);
        if (verified) {
            const uint32_t graph_format_version = 2020051516;

            if (ofi_raw_graph->header()->graph_format_version() < graph_format_version) {
                ENN_ERR_PRINT("CGO graph format version is wrong: correct(%d) != %d\n", graph_format_version,
                              ofi_raw_graph->header()->graph_format_version());
                verified = false;
                return;
            }

            model_mem_info = model;
            cgo_params = params;
            ENN_DBG_PRINT("CGO params size: %zu \n", cgo_params->size());  // TSGD + Param number
        }
    }
    CATCH(what) {
        ENN_ERR_COUT << "CGO model verify failed: " << what << std::endl;
        verified = false;
    }
    END_TRY
}

void CgoParseStrategy::pre_execute() {
    parse_graph_infos_impl();
}

void CgoParseStrategy::parse_graph_infos_impl() {
    std::string name = "";
    std::vector<int32_t> inputs;
    std::vector<int32_t> outputs;

    auto header = ofi_raw_graph->header();
    if (header) {
        if (header->name() != nullptr) {
            name = header->name()->c_str();
        }
    }

    auto graph_core = ofi_raw_graph->core();
    if (graph_core) {
        auto buffers = graph_core->buffers();
        if (buffers) {
            auto in_bufs = graph_core->graph_in_buffers();
            if (in_bufs) {
                for (size_t i = 0; i < in_bufs->size(); ++i) {
                    auto buffer_index = in_bufs->Get(i);
                    auto param_index = buffers->Get(buffer_index)->pool_index();
                    inputs.push_back(param_index);
                }
            }

            auto out_bufs = graph_core->graph_out_buffers();
            if (out_bufs) {
                for (size_t i = 0; i < out_bufs->size(); ++i) {
                    auto buffer_index = out_bufs->Get(i);
                    auto param_index = buffers->Get(buffer_index)->pool_index();
                    outputs.push_back(param_index);
                }
            }
        }
    }

    model_builder.build_graph_info().add_graph_info().set_name(name).set_inputs(inputs).set_outputs(outputs).build();
}

void CgoParseStrategy::parse_operators() {
    auto pre_cpu_core = ofi_raw_graph->pre_cpu_core();
    if (pre_cpu_core) {
        ENN_DBG_PRINT("pre_cpu_core()->size = %d\n", pre_cpu_core->size());

        for (size_t i = 0; i < pre_cpu_core->size(); ++i) {
            parse_operators_impl(pre_cpu_core->Get(i));
        }
    }

    auto graph_core = ofi_raw_graph->core();
    if (graph_core) {
        auto msgs = graph_core->msgs();
        if (msgs) {
            ENN_DBG_PRINT("graph_core->msgs()->size = %d\n", msgs->size());

            for (size_t i = 0; i < msgs->size(); ++i) {
                parse_operators_impl(msgs->Get(i));
            }
        }
    }

    auto post_cpu_core = ofi_raw_graph->post_cpu_core();
    if (post_cpu_core) {
        ENN_DBG_PRINT("post_cpu_core()->size = %d\n", post_cpu_core->size());

        for (size_t i = 0; i < post_cpu_core->size(); ++i) {
            parse_operators_impl(post_cpu_core->Get(i));
        }
    }
}

void CgoParseStrategy::generate_buffer_indexes(const flatbuffers::Vector<uint32_t>* in_container, bool is_scalar,
                                               std::vector<int32_t>* out_container) {
    if (core_buffers == nullptr || core_scalars == nullptr) {
        ENN_ERR_COUT << "core_buffers or core_scalars is not set yet." << std::endl;
        return;
    }

    for (size_t i = 0; i < in_container->size(); ++i) {
        auto index = in_container->Get(i);
        uint32_t param_index = 0;
        if (is_scalar) {
            param_index = core_scalars->Get(index)->pool_index();
            add_new_tensor_name(param_index, core_scalars->Get(index)->name()->c_str());
        } else {
            param_index = core_buffers->Get(index)->pool_index();
            add_new_tensor_name(param_index, core_buffers->Get(index)->name()->c_str());
        }
        if (out_container != nullptr) {
            out_container->push_back(param_index);
        }
    }
}

void CgoParseStrategy::parse_operators_impl(const ofi::rawgraph::fb_OfiMacroSubGraph* msg) {
    uint32_t op_index;
    int32_t op_code = UNDEFINED;
    std::string op_name;
    std::vector<std::string> lib_names;
    std::vector<int32_t> input_indexes;
    std::vector<int32_t> output_indexes;
    std::vector<int32_t> binary_indexes;
    Accelerator accelerator = Accelerator::CUSTOM_CPU_KERNEL;

    core_buffers = ofi_raw_graph->core()->buffers();
    core_scalars = ofi_raw_graph->core()->scalars();

    auto info = msg->kernel_info();
    op_index = msg->msg_id();
    op_code = info->function_id();
    op_name = std::string(get_cpu_kernel_names(op_code));

    switch (msg->assigned_target()) {
        case fb_OfiTargetType_OFI_TARGET_VIP:
        case fb_OfiTargetType_OFI_TARGET_ORCA:
            accelerator = Accelerator::DSP;
            break;
        default:
            accelerator = Accelerator::CUSTOM_CPU_KERNEL;
            break;
    }

    generate_buffer_indexes(msg->out_buffers(), false, &output_indexes);

    if (accelerator == Accelerator::DSP) {
        parse_dsp_binary_impl(lib_names, binary_indexes);

        generate_buffer_indexes(msg->in_buffers(), false, nullptr);
        generate_buffer_indexes(msg->usr_scalars(), true, nullptr);

        auto graph_param = ofi_raw_graph->param();
        if (graph_param) {
            for (int32_t param_index = 0; param_index < graph_param->dev_param_max_idx(); ++param_index) {
                if (!util::is_vector_contain(output_indexes, param_index)) {
                    input_indexes.push_back(param_index);
                }
            }
        }
    } else {
        generate_buffer_indexes(msg->in_buffers(), false, &input_indexes);
        generate_buffer_indexes(msg->usr_scalars(), true, &input_indexes);
    }

    model_builder.build_operator()
        .add_operator()
        .set_op_index(op_index)
        .set_op_code(op_code)
        .set_op_name(op_name)
        .set_lib_names(lib_names)
        .set_input_indexes(input_indexes)
        .set_output_indexes(output_indexes)
        .set_binary_indexes(binary_indexes)
        .set_accelerator(accelerator)
        .build();
}

void CgoParseStrategy::add_new_tensor_name(uint32_t index, std::string name) {
    std::lock_guard<std::mutex> lock_guard_tensor_name_map(mutex_tensor_name_map);
    new_tensor_name_map[index] = name;
}

void CgoParseStrategy::parse_dsp_binary_impl(std::vector<std::string>& lib_names, std::vector<int32_t>& binary_indexes) {
    std::string binary_name;

    auto core_target_info = ofi_raw_graph->core()->target_info();
    if (core_target_info) {
        auto type = core_target_info->type();
        ENN_DBG_COUT << "TSGD Type: " << ofi::rawgraph::EnumNamefb_OfiGraphType(type) << std::endl;
        switch (type) {
            case ofi::rawgraph::fb_OfiGraphType_OFI_GRAPH_TYPE_NN2018: {
                auto ti_dsp2018 = core_target_info->dsp2018();
                binary_name = ti_dsp2018->graph_info()->name()->c_str();
            } break;

            case ofi::rawgraph::fb_OfiGraphType_OFI_GRAPH_TYPE_CVNN2019: {
                auto ti_dsp2019 = core_target_info->dsp2019();
                binary_name = ti_dsp2019->graph_info()->name()->c_str();
                for (auto lib_bin : *ti_dsp2019->lib_bin_list()) {
                    lib_names.push_back(lib_bin->lib_path()->c_str());
                }
            } break;

            default:
                ENN_WARN_COUT << "Invalid TSGD Type: " << type << std::endl;
                break;
        }
    }

    auto dsp_bin = cgo_params->at(0);

    model_builder.build_binary()
        .add_binary()
        .set_index(0)
        .set_name(binary_name)
        .set_address(reinterpret_cast<uint8_t*>(dsp_bin->va))
        .set_fd(dsp_bin->fd)
        .set_offset(dsp_bin->offset)
        .set_size(dsp_bin->size)
        .set_accelerator(Accelerator::DSP)
        .build();

    binary_indexes.push_back(0);
}

bool CgoParseStrategy::validate_operators() {
    // TODO: Implement
    return true;
}

void CgoParseStrategy::parse_tensors() {
    auto graph_infos = model_builder.get_model()->get_graph_infos();
    if (graph_infos.size() == 0) {
        ENN_ERR_COUT << "graph_infos is empty" << std::endl;
        return;
    }

    auto graph_param = ofi_raw_graph->param();
    if (graph_param) {
        auto param_list = graph_param->param_list();
        if (param_list) {
            auto graph_info = graph_infos.at(0);
            constexpr int32_t OFI_BUF_TYPE_U8 = 3;  // TFLite::TensorType_UINT8 = 3

            for (size_t i = 0; i < param_list->size(); ++i) {
                auto param_ele = param_list->Get(i);
                auto buf_info = param_ele->buf_info();
                auto param = cgo_params->at(i + 1);
                uint32_t buffer_size = buf_info->size();
                std::vector<uint32_t> shape = {1, 1, 1, buffer_size};

                bool is_scalar = (param->fd < 0) &&
                                 !util::is_vector_contain(graph_info->get_inputs(), i) &&
                                 !util::is_vector_contain(graph_info->get_outputs(), i);

                if (is_scalar) {
                    model_builder.build_scalar()
                        .add_scalar()
                        .set_index(i)
                        .set_name(buf_info->name()->c_str())
                        .set_type(OFI_BUF_TYPE_U8)
                        .set_shape(shape)
                        .set_address(reinterpret_cast<uint8_t*>(param->va))
                        .set_size(buffer_size)
                        .set_fd(param->fd)
                        .set_offset(param->offset)
                        .build();
                } else {
                    model_builder.build_tensor()
                        .add_tensor()
                        .set_index(i)
                        .set_name(buf_info->name()->c_str())
                        .set_type(OFI_BUF_TYPE_U8)
                        .set_shape(shape)
                        .set_address(reinterpret_cast<uint8_t*>(param->va))
                        .set_size(buffer_size)
                        .build();
                }
            }
        }
    }

    auto buffers = ofi_raw_graph->core()->buffers();
    if (buffers) {
        for (auto buffer : *buffers) {
            std::string buffer_name = buffer->name()->c_str();
            if (buffer_name.find("Shape") != std::string::npos) {
                auto param_index = buffer->pool_index();
                add_new_tensor_name(param_index, buffer_name);
            }
        }
    }
}

bool CgoParseStrategy::validate_tensors() {
    // TODO: Implement
    return true;
}

void CgoParseStrategy::parse_attribute() {
    int32_t graph_version = 0;
    int32_t model_type = (int32_t)ModelType::CGO;

    auto header = ofi_raw_graph->header();
    if (header) {
        graph_version = header->graph_format_version();
    }

    model_builder.build_attribute().add_attribute().set_version(graph_version).set_model_type(model_type).build();
}

bool CgoParseStrategy::validate_attribute() {
    // TODO: Implement
    return true;
}

void CgoParseStrategy::post_execute() {
    auto parsed_model = model_builder.get_model();

    // Set prev/next operation index to each tensor
    for (int op_idx = 0; op_idx < parsed_model->get_operators().size(); op_idx++) {
        auto operator_ = parsed_model->get_operators().at(op_idx);

        if (operator_->get_accelerator() == Accelerator::DSP) {
            auto graph_info = parsed_model->get_graph_infos().at(0);
            auto graph_param = ofi_raw_graph->param();
            if (graph_param) {
                for (int32_t idx = 0; idx < graph_param->dev_param_max_idx(); ++idx) {
                    if (util::is_vector_contain(operator_->get_output_indexes(), idx)) {
                        model_builder.build_tensor().get_tensor_from_index(idx).set_prev_operator_index(op_idx);
                    } else if (!util::is_vector_contain(graph_info->get_outputs(), idx)) {
                        model_builder.build_tensor().get_tensor_from_index(idx).add_next_operator_index(op_idx);
                    }
                }
            }
        } else {
            for (uint32_t idx : operator_->get_input_indexes()) {
                model_builder.build_tensor().get_tensor_from_index(idx).add_next_operator_index(op_idx);
            }
            for (uint32_t idx : operator_->get_output_indexes()) {
                model_builder.build_tensor().get_tensor_from_index(idx).set_prev_operator_index(op_idx);
            }
        }
    }

    // Replace tensor name from param's to buffer/scalar's
    for (auto iter = new_tensor_name_map.begin(); iter != new_tensor_name_map.end(); ++iter) {
        uint32_t index = iter->first;
        std::string name = iter->second;
        model_builder.build_tensor().get_tensor_from_index(index).set_name(name);
    }
}

void CgoParseStrategy::print() {
    auto parsed_model = model_builder.get_model();

    for (size_t gi_idx = 0; gi_idx < parsed_model->get_graph_infos().size(); ++gi_idx) {
        auto graph_info = parsed_model->get_graph_infos().at(gi_idx);
        std::string inputs = " ", outputs = " ";
        for (auto in : graph_info->get_inputs()) {
            inputs += (std::to_string(in) + " ");
        }
        for (auto out : graph_info->get_outputs()) {
            outputs += (std::to_string(out) + " ");
        }
        ENN_DBG_PRINT("Graph Info[[%zu] name : %s, inputs [%s], outputs [%s]\n", gi_idx, graph_info->get_name().c_str(),
                      inputs.c_str(), outputs.c_str());
    }

    for (size_t op_idx = 0; op_idx < parsed_model->get_operators().size(); ++op_idx) {
        auto operator_ = parsed_model->get_operators().at(op_idx);
        ENN_DBG_PRINT("Operator[%2zu] (hw:%2d) (code: %4d) %s\n", op_idx, (int)operator_->get_accelerator(),
                      operator_->get_op_code(), operator_->get_name().c_str());

        std::string inputs = " ";
        std::string outputs = " ";
        std::string binaries = " ";
        for (auto in : operator_->get_input_indexes()) {
            inputs += (std::to_string(in) + " ");
        }
        for (auto out : operator_->get_output_indexes()) {
            outputs += (std::to_string(out) + " ");
        }
        for (auto bin : operator_->get_binary_indexes()) {
            binaries += (std::to_string(bin) + " ");
        }
        ENN_DBG_PRINT("  - In[%s] / Out[%s] / Binary[%s]\n", inputs.c_str(), outputs.c_str(), binaries.c_str());
    }

    for (size_t pa_idx = 0; pa_idx < cgo_params->size(); ++pa_idx) {
        auto param = cgo_params->at(pa_idx);
        std::string str_idx = "TSGD";
        if (pa_idx > 0) {
            str_idx = std::to_string(pa_idx - 1);
        }
        ENN_DBG_PRINT("Parameter[%4s] FD: %d, va: %p, size: %d, offset: %d\n", str_idx.c_str(), param->fd, param->va,
                      param->size, param->offset);
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

        std::string category = "";
        std::string tensor_name = tensor->get_name();
        std::transform(tensor_name.begin(), tensor_name.end(), tensor_name.begin(), ::tolower);
        if (!prev.empty() || tensor_name.find("input") != std::string::npos) {
            category = "(T) ";
        } else if (std::dynamic_pointer_cast<Scalar>(tensor) != nullptr)
            category = "(S) ";
        else {
            category = "(P) ";
        }

        ENN_DBG_PRINT("%6s -> [ %4d ] -> %6s%9d   %s%s\n", prev.c_str(), tensor->get_index(), next.c_str(),
                      tensor->get_size(), category.c_str(), tensor->get_name().c_str());
    }

    for (size_t bidx = 0; bidx < parsed_model->get_binaries().size(); ++bidx) {
        auto binary = parsed_model->get_binaries().at(bidx);
        ENN_DBG_PRINT("Binary[%zu] name: %s, fd: %d, offset: %d, addr: %p, size: %d, accelerator: %d\n", bidx,
                      binary->get_name().c_str(), binary->get_fd(), binary->get_offset(), binary->get_address(),
                      binary->get_size(), static_cast<int>(binary->get_accelerator()));
    }
}

std::shared_ptr<raw::Model> CgoParseStrategy::result() {
    return model_builder.create();
}

};  // namespace model
};  // namespace enn
