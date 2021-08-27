#include "model/generator/generator.hpp"

#include <sys/types.h>
#include <unistd.h>
#include <atomic>
#include <memory>
#include "common/enn_debug.h"
#include "common/identifier.hpp"
#include "model/graph/iterator/methods/topological_sort.hpp"
#include "model/meta_data/buffer_meta_data.hpp"
#include "model/component/operator/operator_builder.hpp"
#include "model/component/tensor/feature_map_builder.hpp"
#include "model/component/tensor/parameter_builder.hpp"
#include "model/component/tensor/scalar_builder.hpp"
#include "medium/enn_medium_utils.hpp"
#include "runtime/client_process/client_process.hpp"


namespace enn {
namespace model {

Generator::Generator() = default;


Model::Ptr Generator::generate_model(std::shared_ptr<raw::Model> const& raw_model,
                                     const runtime::ClientProcess::Ptr& client_process) {
    if (raw_model == nullptr) {
#ifndef VELOCE_SOC
        throw std::invalid_argument("Raw Model is Null");
#endif
    }
    ENN_DBG_COUT << "Generate ENN Model Start" << std::endl;

    // 1. Generate Tensors
    generate_tensor_map(raw_model);

    // 2. Generate Operators
    generate_operator_vector(raw_model);

    // 3. Generate Graph using above components.
    generate_graph(raw_model);

    // 4. Generate Model by graph
    this->enn_model = std::make_shared<Model>(client_process, graph);

    // 5. Generate Attribute
    // TODO(yc18.cho): set attribute from real raw_model from parser.(for now nothing in body)
    generate_attribute(raw_model);

    // 6. Generate Buffer Meta Data for Memory Allocation.
    generate_buffer_meta_data(raw_model);

    ENN_DBG_COUT << "Generate ENN Model Completed" << std::endl;
    return std::move(this->enn_model);
}

bool is_input_of_model(std::vector<std::shared_ptr<raw::data::GraphInfo>> const& raw_graph_infos, int32_t tensor_id) {
    bool ret = false;

    for (auto& raw_graph_info : raw_graph_infos) {
        auto& input_vector = raw_graph_info->get_inputs();
        if (std::find(input_vector.begin(), input_vector.end(), tensor_id) != input_vector.end()) {
            ret = true;
            break;
        }
    }

    return ret;
}

// TODO(yc18.cho&GraphGen, TBD): It'll be removed after the tensor's type is added in NNC.
FeatureMap::Type get_feature_map_type(std::vector<std::shared_ptr<raw::data::GraphInfo>> const& raw_graph_infos,
                                        int32_t tensor_id) {
    for (auto& raw_graph_info : raw_graph_infos) {
        if (std::find(raw_graph_info->get_inputs().begin(), raw_graph_info->get_inputs().end(), tensor_id)
                    != raw_graph_info->get_inputs().end()) {
            return FeatureMap::Type::SUBGRAPH_INPUT;
        } else if (std::find(raw_graph_info->get_outputs().begin(), raw_graph_info->get_outputs().end(), tensor_id)
                    != raw_graph_info->get_outputs().end()) {
            return FeatureMap::Type::SUBGRAPH_OUTPUT;
        }
    }
    return FeatureMap::Type::INTERMEDIATE;
}

void Generator::generate_tensor_map(std::shared_ptr<raw::Model> const& raw_model) {
    ENN_DBG_COUT << "Generate Tensors for Enn Model" << std::endl;
    component::FeatureMapBuilder featuremap_builder;
    component::ParameterBuilder parameter_builder;
    component::ScalarBuilder scalar_builder;

    for (auto& raw_tensor : raw_model->get_tensors()) {
        // TODO(Byungjin): No vector in prev
        if ((raw_tensor->get_prev_operator_index() != UNDEFINED) ||
            is_input_of_model(raw_model->get_graph_infos(), raw_tensor->get_index())) {  // Featuremap Case
            tensor_map[raw_tensor->get_index()] =
                featuremap_builder.set_id(raw_tensor->get_index())
                                   .set_name(raw_tensor->get_name())
                                   .set_shape(raw_tensor->get_shape())
                                   .set_data_type(static_cast<TFlite::TensorType>(raw_tensor->get_type()))
                                   .set_buffer_index(raw_tensor->get_index())
                                   .set_buffer_size(raw_tensor->get_size())
                                   .set_quantization_parameters(raw_tensor->get_quantization_parameters())
                                   .set_type(get_feature_map_type(raw_model->get_graph_infos(), raw_tensor->get_index()))
                                   .set_symm_per_channel_quant_parameters(raw_tensor->get_symm_per_channel_quant_parameters())
                                   .create();
        } else if (std::dynamic_pointer_cast<raw::data::Scalar>(raw_tensor) != nullptr) {  // Scalar Case
            tensor_map[raw_tensor->get_index()] =
              scalar_builder.set_id(raw_tensor->get_index())
                            .set_name(raw_tensor->get_name())
                            .set_shape(raw_tensor->get_shape())
                            .set_indexed_buffer_index(raw_tensor->get_index())
                            .set_indexed_buffer_size(raw_tensor->get_size())
                            .set_default_buffer_addr(raw_tensor->get_address())
                            .set_default_buffer_size(raw_tensor->get_size())
                            // .set_default_buffer_fd()
                            // .set_default_buffer_offset()
                            .set_data_type(static_cast<TFlite::TensorType>(raw_tensor->get_type()))
                            .create();
        } else {  // Parameter Case
            tensor_map[raw_tensor->get_index()] =
                parameter_builder.set_id(raw_tensor->get_index())
                                 .set_name(raw_tensor->get_name())
                                 // .set_next_operator(raw_tensor->get_next_operator_index())
                                 .set_shape(raw_tensor->get_shape())
                                 .set_data_type(static_cast<TFlite::TensorType>(raw_tensor->get_type()))
                                 // .set_buffer_fd(raw_tensor)
                                 .set_buffer_addr(raw_tensor->get_address())
                                 .set_buffer_size(raw_tensor->get_size())
                                 // .set_buffer_offset()
                                 .set_quantization_parameters(raw_tensor->get_quantization_parameters())
                                 .set_symm_per_channel_quant_parameters(raw_tensor->get_symm_per_channel_quant_parameters())
                                 .create();
        }
        ENN_DBG_COUT << "-Tensor [ID : " << tensor_map[raw_tensor->get_index()]->get_id()
                     << "][Name : " << tensor_map[raw_tensor->get_index()]->get_name() << "] Created." << std::endl;
    }
    ENN_DBG_COUT << "Generate Tensors Completed" << std::endl;
}

inline void Generator::set_binary_to_op_builder(component::OperatorBuilder& op_builder,
                                                std::vector<std::shared_ptr<raw::data::Binary>> const& raw_binaries,
                                                std::vector<int>& indexes) {
    for (auto& index : indexes) {
        if (index < 0) {
            continue;
        }
        auto& binary = raw_binaries.at(index);

        // ToDo(jungho7.kim): Remove this condition after changing to use fd in NPU UD
        if (available_accelerator(op_builder.get()->get_accelerator(), Accelerator::NPU)) {
            op_builder.add_binary(binary->get_name(), static_cast<const void*>(binary->get_address() + binary->get_offset()),
                                  binary->get_size(), binary->get_accelerator());
        } else {
            op_builder.add_binary(binary->get_name(), binary->get_fd(), static_cast<const void*>(binary->get_address()),
                                  binary->get_size(), binary->get_offset(), binary->get_accelerator());
        }
        ENN_DBG_PRINT("Binary[%d] name: %s, fd: %d, offset: %d, addr: %p, size: %d, hw: %d \n", index,
                      binary->get_name().c_str(), binary->get_fd(), binary->get_offset(), binary->get_address(),
                      binary->get_size(), (int)binary->get_accelerator());
    }
}

inline void Generator::set_option_to_op_builder(component::OperatorBuilder& op_builder,
                                                std::vector<std::shared_ptr<raw::data::OperatorOptions>> const& raw_options,
                                                int index) {
    if (index >= 0 && index < raw_options.size()) {
        auto& operator_option = raw_options.at(index);

        op_builder.set_option(
            operator_option->get_name(),
            operator_option->get_options(),
            0, /*TODO(yc18.cho): Delete Size*/
            static_cast<TFlite::BuiltinOptions>(operator_option->get_number()));
    } else {
        ENN_DBG_COUT << "Invalid index of raw_options to set : " << index << std::endl;
    }
}

inline void Generator::set_npu_option_to_op_builder(
    component::OperatorBuilder& op_builder, std::vector<std::shared_ptr<raw::data::NPUOptions>>& raw_npu_option_vector) {
    if (raw_npu_option_vector.size() > 0) {
        ENN_DBG_COUT << "There is " << raw_npu_option_vector.size() << " npu option in model"  << std::endl;
        op_builder.set_ifm_bound(raw_npu_option_vector.at(0)->is_binding_ifm());
        op_builder.set_ofm_bound(raw_npu_option_vector.at(0)->is_binding_ofm());
    } else {
        ENN_DBG_COUT << "There is no npu option in model : " << raw_npu_option_vector.size() << std::endl;
    }
}

inline void Generator::set_dsp_option_to_op_builder(
    component::OperatorBuilder& op_builder, std::vector<std::shared_ptr<raw::data::DSPOptions>>& raw_dsp_option_vector) {
    if (raw_dsp_option_vector.size() > 0) {
        ENN_DBG_COUT << "There is " << raw_dsp_option_vector.size() << " dsp option in model"  << std::endl;
        // todo(daewhan.kim) : control when there are multiple dsp options in vector.
        ENN_DBG_COUT << "DSP Async Exec : " << raw_dsp_option_vector.at(0)->is_async_exec() << std::endl;
        ENN_DBG_COUT << "DSP Binding IFM : " << raw_dsp_option_vector.at(0)->is_binding_ifm() << std::endl;
        ENN_DBG_COUT << "DSP Binding OFM : " << raw_dsp_option_vector.at(0)->is_binding_ofm() << std::endl;
        op_builder.set_dsp_async_exec(raw_dsp_option_vector.at(0)->is_async_exec());
        op_builder.set_ifm_bound(raw_dsp_option_vector.at(0)->is_binding_ifm());
        op_builder.set_ofm_bound(raw_dsp_option_vector.at(0)->is_binding_ofm());
    } else {
        ENN_DBG_COUT << "There is no dsp option in model : " << raw_dsp_option_vector.size() << std::endl;
    }
}

void Generator::generate_operator_vector(std::shared_ptr<raw::Model> const& raw_model) {
    ENN_DBG_COUT << "Generate Operators for Enn Model" << std::endl;
    auto& raw_operators = raw_model->get_operators();
    auto& raw_binaries = raw_model->get_binaries();
    auto& raw_operators_options = raw_model->get_operator_options();

    component::OperatorBuilder operator_builder;

    for (auto& raw_operator : raw_operators) {
        auto& op_builder = operator_builder.set_id(raw_operator->get_index())
                                           .set_name(raw_operator->get_name())
                                           .set_accelerator(raw_operator->get_accelerator())
                                           .set_code((TFlite::BuiltinOperator)raw_operator->get_op_code())
                                           .set_lib_names(raw_operator->get_lib_names());

        set_binary_to_op_builder(op_builder, raw_binaries, raw_operator->get_binary_indexes());

        set_option_to_op_builder(op_builder, raw_operators_options, raw_operator->get_operator_options_index());

        if (available_accelerator(raw_operator->get_accelerator(), Accelerator::NPU)) {
            set_npu_option_to_op_builder(op_builder, raw_model->get_npu_options());
        } else if (available_accelerator(raw_operator->get_accelerator(), Accelerator::DSP)) {
            set_dsp_option_to_op_builder(op_builder, raw_model->get_dsp_options());
        }

        operator_vector.push_back(op_builder.create());

        ENN_DBG_COUT << "-Operator [ID : " << operator_vector.back()->get_id()
                     << "][Name : " << operator_vector.back()->get_name() << "] Created." << std::endl;
    }

    // TODO(daewhan.kim) : figure out below usage.
    // raw_operator->get_ofm_indexes;
    // raw_operator->get_scalar_indexes();
    ENN_DBG_COUT << "Generate Operators Completed" << std::endl;
}

inline void Generator::connect_tensor_to_next_op(component::Tensor::Ptr& tensor, component::Operator::Ptr& next_op) {
    component::OperatorBuilder operator_builder{next_op};
    if (std::dynamic_pointer_cast<component::Scalar>(tensor) != nullptr) {
        // Scalar Case
        auto scalar = std::static_pointer_cast<component::Scalar>(tensor);
        component::ScalarBuilder scalar_builder{scalar};

        scalar_builder.add_next_operator(next_op);
        operator_builder.add_in_tensor(scalar);

    } else if (tensor->is_const()) {
        // Parameter Case
        auto parameter = std::static_pointer_cast<component::Parameter>(tensor);
        component::ParameterBuilder parameter_builder{parameter};

        parameter_builder.add_next_operator(next_op);
        operator_builder.add_in_tensor(parameter);

    } else {
        // FeatureMap Case
        auto featuremap = std::static_pointer_cast<component::FeatureMap>(tensor);
        component::FeatureMapBuilder feature_map_builder{featuremap};

        feature_map_builder.add_next_operator(next_op);
        operator_builder.add_in_tensor(featuremap);
    }
}

inline void Generator::connect_tensor_to_prev_op(component::Tensor::Ptr& tensor, component::Operator::Ptr& prev_op) {
    component::OperatorBuilder operator_builder{prev_op};
    auto featuremap = std::static_pointer_cast<component::FeatureMap>(tensor);
    component::FeatureMapBuilder feature_map_builder{featuremap};

    feature_map_builder.set_prev_operator(prev_op);
    operator_builder.add_out_tensor(featuremap);
}

inline void Generator::set_virtual_in_out_node(std::vector<std::shared_ptr<raw::data::GraphInfo>> const& raw_graph_infos) {
    ENN_DBG_COUT << "Generate Virtual In/Out" << std::endl;
    component::OperatorBuilder operator_builder;

    component::Operator::Ptr v_in = operator_builder.set_id(-1)
                                                    .set_name(static_cast<std::string>("virtual_input"))
                                                    .create();

    component::Operator::Ptr v_out = operator_builder.set_id(-1)
                                                     .set_name(static_cast<std::string>("virtual_output"))
                                                     .create();

    this->graph->add_vertex(v_in).set_start_vertex(v_in);
    this->graph->add_vertex(v_out).set_end_vertex(v_out);

    for (auto& raw_graph_info : raw_graph_infos) {
        for (auto& input : raw_graph_info->get_inputs()) {
            auto in_tensor = tensor_map[input];
            connect_tensor_to_prev_op(in_tensor, v_in);
            for (auto& next_op : in_tensor->next()) {
                this->graph->add_neighbor(v_in,
                                          std::static_pointer_cast<component::FeatureMap>(in_tensor),
                                          std::static_pointer_cast<component::Operator>(next_op));
            }
        }

        for (auto& output : raw_graph_info->get_outputs()) {
            auto out_tensor = tensor_map[output];
            connect_tensor_to_next_op(out_tensor, v_out);
            this->graph->add_neighbor(std::static_pointer_cast<component::Operator>(out_tensor->prev()),
                                      std::static_pointer_cast<component::FeatureMap>(out_tensor),
                                      v_out);
        }
    }
    ENN_DBG_COUT << "Generate Virtual In/Out Completed" << std::endl;
}

void Generator::generate_graph(std::shared_ptr<raw::Model> const& raw_model) {
    ENN_DBG_COUT << "Generate Original Graph for Enn Model" << std::endl;
    auto& raw_graph_infos = raw_model->get_graph_infos();
    auto& raw_tensors = raw_model->get_tensors();
    auto& raw_operators = raw_model->get_operators();

    this->graph = std::make_shared<OriginalGraph>();

    // Operator & Tensor Setting.
    for (auto& raw_op : raw_operators) {

        component::Operator::Ptr op = operator_vector[raw_op->get_index()];
        this->graph->add_vertex(op);

        for (auto tensor_id : raw_op->get_input_indexes()) {
            connect_tensor_to_next_op(tensor_map[tensor_id], op);
        }

        for (auto tensor_id : raw_op->get_output_indexes()) {
            connect_tensor_to_prev_op(tensor_map[tensor_id], op);
        }
    }

    // Graph Setting.
    for (uint i = 0; i < raw_tensors.size(); i++) {
        int prev_op_index = raw_tensors[i]->get_prev_operator_index();
        auto& next_op_index_vector = raw_tensors[i]->get_next_operator_indexes();
        int32_t tensor_key = raw_tensors[i]->get_index();
        if (!tensor_map[tensor_key]->is_const()
            && std::dynamic_pointer_cast<component::Scalar>(tensor_map[tensor_key]) == nullptr) {
            auto edge = std::static_pointer_cast<component::FeatureMap>(tensor_map[tensor_key]);
            component::Operator::Ptr prev_op, next_op;

            // Set Prev Operator
            if (prev_op_index >= 0) {
                prev_op = operator_vector[prev_op_index];
            }

            // Set Next Operators
            for (int next_op_index : next_op_index_vector) {
                if (next_op_index >= 0) {
                    next_op = operator_vector[next_op_index];
                }

                // Add Connection to Graph's adjacency list
                if (prev_op != nullptr && next_op != nullptr)
                    this->graph->add_neighbor(prev_op, edge, next_op);
            }
        }
    }

    // Set Virtual In / Out Vertex
    set_virtual_in_out_node(raw_graph_infos);
    ENN_DBG_COUT << "Generate Original Graph Completed" << std::endl;
}

inline void Generator::input_bind(component::Operator::Ptr& opr_ptr,
                                  uint32_t& binding_ifm_count, uint32_t& region_index) {
    // operator's in_tensors can be 3 types (FeatureMap, Scalar, Parameter).
    // For binidng Input, need to count only FeatureMap.
    for (auto& in_tensor : opr_ptr->in_tensors) {
        if (!in_tensor->is_const()) {  // Feature Map
            ++binding_ifm_count;

            auto& buf_meta_data = this->enn_model->get_buffer_meta_data().at(in_tensor->get_id());
            if (buf_meta_data->get_region_index() < region_index) {
                region_index = buf_meta_data->get_region_index();
            }
        }
    }
    ENN_DBG_COUT << "Binding : " << binding_ifm_count
                 << " IFMs to 1, Below buffer meta data changed" << std::endl;

    // Change the buffer meta data which are made by pre operator's output.
    for (auto& in_tensor : opr_ptr->in_tensors) {
        if (!in_tensor->is_const()) {  // Feature Map
            auto& meta_data_to_change = this->enn_model->get_buffer_meta_data().at(in_tensor->get_id());
            meta_data_to_change->set_region_index(region_index);

            meta_data_to_change->print();
        }
    }
    region_index++;

    component::OperatorBuilder operator_builder{opr_ptr};
    operator_builder.set_ifm_bound(true);
}

inline void Generator::output_bind(component::Operator::Ptr& opr_ptr,
                                   metadata::BufferMetaData::Ptr& buffer_meta_data,
                                   uint32_t& binding_ofm_count, uint32_t& region_index) {
    if (binding_ofm_count == 0) {
        ENN_DBG_COUT << "Binding " << opr_ptr->out_tensors.count() << " OFMs to 1" << std::endl;
    }
    buffer_meta_data->set_region_index(region_index);
    ++binding_ofm_count;
    if (binding_ofm_count == opr_ptr->out_tensors.count()) {
        ++region_index;
    }

    component::OperatorBuilder operator_builder{opr_ptr};
    operator_builder.set_ofm_bound(true);
}

inline bool is_tensor_between_gpu_ops(const component::FeatureMap::Ptr& fm) {
    const auto& prev_op = std::static_pointer_cast<component::Operator>(fm->prev());
    if (prev_op->get_accelerator() != Accelerator::GPU)
        return false;

    for (auto& next : fm->next()) {
        const auto& next_op = std::static_pointer_cast<component::Operator>(next);
        if (next_op->get_accelerator() != Accelerator::GPU)
            return false;
    }

    return true;
}

void Generator::generate_buffer_meta_data(std::shared_ptr<raw::Model> const& raw_model) {
    ENN_DBG_COUT << "Generate Buffer Meta Data" << std::endl;
    std::vector<int32_t> input_ids;
    for (auto& in_tensor : this->graph->get_start_vertex()->out_tensors) {
        // Push only FeatureMap that is not constant among FeatureMap and Parameter.
        if (!in_tensor->is_const()) input_ids.push_back(in_tensor->get_id());
    }

    std::vector<int32_t> output_ids;
    for (auto& out_tensor : this->graph->get_end_vertex()->in_tensors) {
        output_ids.push_back(out_tensor->get_id());
    }

    std::vector<uint32_t> direction_index = {0, 0, 0};
    uint32_t region_index = 0;
    uint32_t binding_ofm_count = 0;
    uint32_t binding_ifm_count = 0;
    for (auto& opr_ptr : this->enn_model->get_origin_graph()->order<enn::model::graph::iterator::TopologicalSort>()) {
        if ((opr_ptr->get_accelerator() == Accelerator::NPU) && opr_ptr->is_ifm_bound()) {
            input_bind(opr_ptr, binding_ifm_count, region_index);
        }

        ENN_DBG_COUT << "Operator(" << (int)opr_ptr->get_id() << ") Needs below Tenosr(S) Alloc" << std::endl;
        for (auto& in_tensor : opr_ptr->in_tensors) {
            auto scalar_tensor = std::dynamic_pointer_cast<component::Scalar>(in_tensor);
            if (scalar_tensor != nullptr) {
                metadata::BufferMetaData::Ptr buffer_meta_data = std::make_shared<metadata::BufferMetaData>();
                buffer_meta_data->set_name(scalar_tensor->get_name())
                                ->set_index(scalar_tensor->get_indexed_buffer_index())
                                ->set_shape(scalar_tensor->get_shape())
                                ->set_data_type(scalar_tensor->get_data_type())
                                ->set_size(scalar_tensor->get_indexed_buffer_size())
                                ->set_direction(Direction::EXT)
                                ->set_direction_index(direction_index[static_cast<uint32_t>(Direction::EXT)]++)
                                ->set_region_index(region_index);

                this->enn_model->add_buffer_meta_data(buffer_meta_data);
                buffer_meta_data->print();
                ++region_index;
            }
        }
        ENN_DBG_COUT << "Operator(" << (int)opr_ptr->get_id() << ") Needs below Tensor(F) Alloc" << std::endl;
        for (auto& out_tensor : opr_ptr->out_tensors) {
            // parameter & scalar cannot have prev-operator, so out tesnors are always featuremap-type tensor.
            auto fm = std::static_pointer_cast<component::FeatureMap>(out_tensor);

            // gpu user driver will allocate for feature map between gpu operators,
            // Therefore, skip request memory allocation for feature maps betwwen gpu operators.
            if (is_tensor_between_gpu_ops(fm)) {
                continue;
            }

            // TODO(daewhan.kim) : Remove Below Code
            //                     Below codes are for DLV3 model,
            //                     When There is npu, dsp operation seperately,
            //                     npu output and dsp output are Same.
            //                     Tensor Cannot Have multi Prev(IN) Operator,
            //                     But for now, just skip the DSP's output tensor alloc,
            //                     if dsp output tensor is equal with npu output tensor.
            {  // HACK CODE
                if ((opr_ptr->get_accelerator() == Accelerator::DSP)) {
                    auto& raw_operators = raw_model->get_operators();
                    int32_t npu_out_tensor_index = UNDEFINED;
                    int32_t dsp_out_tensor_index = UNDEFINED;
                    for (auto& raw_op : raw_operators) {
                        if (raw_op->get_accelerator() == Accelerator::NPU) {
                            npu_out_tensor_index = raw_op->get_output_indexes().at(util::FIRST);
                        } else if (raw_op->get_accelerator() == Accelerator::DSP) {
                            dsp_out_tensor_index = raw_op->get_output_indexes().at(util::FIRST);
                        }
                    }

                    if (npu_out_tensor_index != UNDEFINED && dsp_out_tensor_index != UNDEFINED) {
                        if (npu_out_tensor_index == dsp_out_tensor_index) {
                            ENN_DBG_COUT << "NPU and DSP have same output tensor, SKIP generate DSP output buffer meta data."
                                        << std::endl;
                            ENN_DBG_COUT << "This Buffer Meta Data will be generated when generate NPU output buffer meta data."
                                        << std::endl;
                            continue;
                        }
                    }
                }
            }
            metadata::BufferMetaData::Ptr buffer_meta_data = std::make_shared<metadata::BufferMetaData>();
            buffer_meta_data->set_name(fm->get_name())
                            ->set_index(fm->get_buffer_index())
                            ->set_shape(fm->get_shape())
                            ->set_data_type(fm->get_data_type())
                            ->set_size(fm->get_buffer_size());

            if (std::find(input_ids.begin(), input_ids.end(), fm->get_id()) != input_ids.end()) {
                buffer_meta_data->set_direction(Direction::Input)
                                ->set_direction_index(direction_index[static_cast<uint32_t>(Direction::Input)]++);
            } else if (std::find(output_ids.begin(), output_ids.end(), fm->get_id()) != output_ids.end()) {
                buffer_meta_data->set_direction(Direction::Output)
                                ->set_direction_index(direction_index[static_cast<uint32_t>(Direction::Output)]++);
            } else {
                buffer_meta_data->set_direction(Direction::EXT)
                                ->set_direction_index(direction_index[static_cast<uint32_t>(Direction::EXT)]++);
            }

            if ((opr_ptr->get_accelerator() == Accelerator::NPU || opr_ptr->get_accelerator() == Accelerator::DSP)
                && (opr_ptr->is_ofm_bound())) {
                output_bind(opr_ptr, buffer_meta_data, binding_ofm_count, region_index);
            } else {
                buffer_meta_data->set_region_index(region_index);
                ++region_index;
            }

            this->enn_model->add_buffer_meta_data(buffer_meta_data);
            buffer_meta_data->print();
        }
    }
    ENN_DBG_COUT << "Generate Buffer Meta Data Completed" << std::endl;
}

void Generator::generate_attribute(std::shared_ptr<raw::Model> const &raw_model) {
    // TODO(UGG+UENN): set NNAPI_TYPE in nnc
    auto model_options = raw_model->get_model_options();

    if (model_options == nullptr) {
        ENN_DBG_COUT << "There is no model option to Set" << std::endl;
        return;
    }

    ENN_DBG_COUT << "generate Attribute" << std::endl;
    this->enn_model->set_attribute(
        std::make_shared<Attribute>()
            ->set_lagacy_model(static_cast<TFlite::LegacyModel>(model_options->get_legacy_model()))
            ->set_relax_computation_float32_to_float16(model_options->is_relax_computation_float32_to_float16()));
}

};  // namespace model
};  // namespace enn

