#include "model/raw/data/attribute.hpp"
#include "model/raw/data/binary.hpp"
#include "model/raw/data/buffer.hpp"
#include "model/raw/data/control_option.hpp"
#include "model/raw/data/graph_info.hpp"
#include "model/raw/data/model_option.hpp"
#include "model/raw/data/npu_options.hpp"
#include "model/raw/data/dsp_options.hpp"
#include "model/raw/data/operator.hpp"
#include "model/raw/data/operator_options.hpp"
#include "model/raw/data/region.hpp"
#include "model/raw/data/scalar.hpp"
#include "model/raw/data/tensor.hpp"

namespace enn {
namespace model {
namespace raw {

/*
 *
 */
Model::Model() = default;

Model::~Model() = default;

std::vector<std::shared_ptr<data::Binary>>& Model::get_binaries() {
    return binaries_;
}

std::vector<std::shared_ptr<data::Buffer>>& Model::get_buffers() {
    return buffers_;
}

std::vector<std::shared_ptr<data::Operator>>& Model::get_operators() {
    return operators_;
}

std::vector<std::shared_ptr<data::OperatorOptions>>& Model::get_operator_options() {
    return operator_options_;
}

std::vector<std::shared_ptr<data::Region>>& Model::get_regions() {
    return regions_;
}

std::vector<std::shared_ptr<data::Scalar>>& Model::get_scalars() {
    return scalars_;
}

std::vector<std::shared_ptr<data::Tensor>>& Model::get_tensors() {
    return tensors_;
}

std::vector<std::shared_ptr<data::GraphInfo>>& Model::get_graph_infos() {
    return graph_infos_;
}

std::vector<std::shared_ptr<data::NPUOptions>>& Model::get_npu_options() {
    return npu_options_;
}

std::vector<std::shared_ptr<data::DSPOptions>>& Model::get_dsp_options() {
    return dsp_options_;
}

std::shared_ptr<data::ModelOption> Model::get_model_options() {
    return model_options_;
}

std::shared_ptr<data::Attribute> Model::get_attribute() {
    return attribute_;
}

std::shared_ptr<data::ControlOption> Model::get_control_option() {
    return control_option_;
}

/*
 *
 */
ModelBuilder::~ModelBuilder() = default;

data::BinaryBuilder ModelBuilder::build_binary() const {
    return data::BinaryBuilder(raw_model_);
}

data::BufferBuilder ModelBuilder::build_buffer() const {
    return data::BufferBuilder(raw_model_);
}

data::ModelOptionBuilder ModelBuilder::build_model_option() const {
    return data::ModelOptionBuilder(raw_model_);
}

data::OperatorBuilder ModelBuilder::build_operator() const {
    return data::OperatorBuilder(raw_model_);
}

data::OperatorOptionsBuilder ModelBuilder::build_operator_options() const {
    return data::OperatorOptionsBuilder(raw_model_);
}

data::RegionBuilder ModelBuilder::build_region() const {
    return data::RegionBuilder(raw_model_);
}

data::ScalarBuilder ModelBuilder::build_scalar() const {
    return data::ScalarBuilder(raw_model_);
}

data::TensorBuilder ModelBuilder::build_tensor() const {
    return data::TensorBuilder(raw_model_);
}

data::GraphInfoBuilder ModelBuilder::build_graph_info() const {
    return data::GraphInfoBuilder(raw_model_);
}

data::AttributeBuilder ModelBuilder::build_attribute() const {
    return data::AttributeBuilder(raw_model_);
}

data::ControlOptionBuilder ModelBuilder::build_control_option() const {
    return data::ControlOptionBuilder(raw_model_);
}

data::NPUOptionsBuilder ModelBuilder::build_npu_options() const {
    return data::NPUOptionsBuilder(raw_model_);
}

data::DSPOptionsBuilder ModelBuilder::build_dsp_options() const {
    return data::DSPOptionsBuilder(raw_model_);
}

};  // namespace raw
};  // namespace model
};  // namespace enn
