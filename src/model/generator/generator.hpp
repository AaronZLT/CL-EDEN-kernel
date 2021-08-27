#ifndef MODEL_MANAGER_MANAGER_HPP_
#define MODEL_MANAGER_MANAGER_HPP_

#include <map>
#include <vector>
#include <string>

#include "model/model.hpp"
#include "model/graph/graph.hpp"
#include "model/component/tensor/feature_map.hpp"
#include "model/component/tensor/parameter.hpp"
#include "model/component/tensor/scalar.hpp"
#include "model/raw/model.hpp"
#include "model/raw/data/operator.hpp"
#include "model/raw/data/tensor.hpp"
#include "model/raw/data/binary.hpp"
#include "model/raw/data/operator_options.hpp"
#include "model/raw/data/graph_info.hpp"
#include "model/raw/data/npu_options.hpp"
#include "model/raw/data/dsp_options.hpp"
#include "model/raw/data/model_option.hpp"
#include "model/graph_types.hpp"
#include "model/raw/data/buffer.hpp"
#include "model/raw/data/scalar.hpp"
#include "model/raw/data/attribute.hpp"
#include "runtime/client_process/client_process.hpp"

namespace enn {
namespace model {

namespace component {
class Operator;
class OperatorBuilder;
}

class Generator {
 public:
    explicit Generator();

    Model::Ptr generate_model(std::shared_ptr<raw::Model> const& raw_model,
                              const runtime::ClientProcess::Ptr& client_process);

 private:
    void generate_buffer_meta_data(std::shared_ptr<raw::Model> const& raw_model);

    void generate_tensor_map(std::shared_ptr<raw::Model> const& raw_model);

    void generate_feature_map_vector(std::shared_ptr<raw::Model> const& raw_model);

    void generate_operator_vector(std::shared_ptr<raw::Model> const& raw_model);

    void generate_graph(std::shared_ptr<raw::Model> const& raw_model);

    void generate_attribute(std::shared_ptr<raw::Model> const& raw_model);

    inline void set_virtual_in_out_node(std::vector<std::shared_ptr<raw::data::GraphInfo>> const& raw_graph_infos);

    inline void set_binary_to_op_builder(component::OperatorBuilder& op_builder,
        std::vector<std::shared_ptr<raw::data::Binary>> const& raw_binaries, std::vector<int>& indexes);

    inline void set_option_to_op_builder(component::OperatorBuilder& op_builder,
        std::vector<std::shared_ptr<raw::data::OperatorOptions>> const& raw_options, int index);

    inline void set_dsp_option_to_op_builder(component::OperatorBuilder& op_builder,
        std::vector<std::shared_ptr<raw::data::DSPOptions>>& raw_dsp_option_vector);

    inline void set_npu_option_to_op_builder(component::OperatorBuilder& op_builder,
        std::vector<std::shared_ptr<raw::data::NPUOptions>>& raw_npu_option_vector);

    inline void connect_tensor_to_next_op(component::Tensor::Ptr& tensor, component::Operator::Ptr& next_op);

    inline void connect_tensor_to_prev_op(component::Tensor::Ptr& tensor, component::Operator::Ptr& prev_op);

    inline void input_bind(component::Operator::Ptr& opr_ptr, uint32_t& binding_ifm_count, uint32_t& region_index);

    inline void output_bind(component::Operator::Ptr& opr_ptr, metadata::BufferMetaData::Ptr& buffer_meta_data,
        uint32_t& binding_ofm_count, uint32_t& region_index);

    Model::Ptr enn_model;
    std::unordered_map<int32_t, component::Tensor::Ptr> tensor_map;
    std::vector<component::Operator::Ptr> operator_vector;
    OriginalGraph::Ptr graph;
};


};  // namespace model
};  // namespace enn

#endif  // MODEL_MANAGER_MANAGER_HPP_
