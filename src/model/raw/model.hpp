#ifndef SRC_MODEL_RAW_MODEL_HPP_
#define SRC_MODEL_RAW_MODEL_HPP_

#include <memory>
#include <string>
#include <vector>

namespace enn {
namespace model {

enum class ModelType : uint32_t { NONE, NNC, CGO, CODED, SIZE };

class ModelMemInfo {
public:
    ModelMemInfo(void* va, int32_t fd, int32_t size = 0, int32_t offset = 0) {
        this->va = va;
        this->fd = fd;
        this->size = size;
        this->offset = offset;
    }

    void* va;
    int32_t fd;
    int32_t size;
    int32_t offset;
};

namespace raw {

namespace data {
class Attribute;
class Binary;
class Buffer;
class ControlOption;
class GraphInfo;
class ModelOption;
class NPUOptions;
class DSPOptions;
class Operator;
class OperatorOptions;
class Region;
class Scalar;
class Tensor;

class AttributeBuilder;
class BinaryBuilder;
class BufferBuilder;
class ControlOptionBuilder;
class GraphInfoBuilder;
class ModelOptionBuilder;
class NPUOptionsBuilder;
class DSPOptionsBuilder;
class OperatorBuilder;
class OperatorOptionsBuilder;
class RegionBuilder;
class ScalarBuilder;
class TensorBuilder;
};  // namespace data

class Model {
private:
    std::vector<std::shared_ptr<data::Binary>> binaries_;
    std::vector<std::shared_ptr<data::Buffer>> buffers_;
    std::vector<std::shared_ptr<data::GraphInfo>> graph_infos_;
    std::vector<std::shared_ptr<data::Operator>> operators_;
    std::vector<std::shared_ptr<data::OperatorOptions>> operator_options_;
    std::vector<std::shared_ptr<data::Region>> regions_;
    std::vector<std::shared_ptr<data::Scalar>> scalars_;
    std::vector<std::shared_ptr<data::Tensor>> tensors_;
    std::vector<std::shared_ptr<data::NPUOptions>> npu_options_;
    std::vector<std::shared_ptr<data::DSPOptions>> dsp_options_;
    std::shared_ptr<data::ModelOption> model_options_;
    std::shared_ptr<data::Attribute> attribute_;
    std::shared_ptr<data::ControlOption> control_option_;

    friend class data::AttributeBuilder;
    friend class data::BinaryBuilder;
    friend class data::BufferBuilder;
    friend class data::ControlOptionBuilder;
    friend class data::GraphInfoBuilder;
    friend class data::ModelOptionBuilder;
    friend class data::NPUOptionsBuilder;
    friend class data::DSPOptionsBuilder;
    friend class data::OperatorBuilder;
    friend class data::OperatorOptionsBuilder;
    friend class data::RegionBuilder;
    friend class data::ScalarBuilder;
    friend class data::TensorBuilder;

public:
    Model();
    ~Model();

    std::vector<std::shared_ptr<data::Binary>>& get_binaries();
    std::vector<std::shared_ptr<data::Buffer>>& get_buffers();
    std::vector<std::shared_ptr<data::GraphInfo>>& get_graph_infos();
    std::vector<std::shared_ptr<data::Operator>>& get_operators();
    std::vector<std::shared_ptr<data::OperatorOptions>>& get_operator_options();
    std::vector<std::shared_ptr<data::Region>>& get_regions();
    std::vector<std::shared_ptr<data::Scalar>>& get_scalars();
    std::vector<std::shared_ptr<data::Tensor>>& get_tensors();
    std::vector<std::shared_ptr<data::NPUOptions>>& get_npu_options();
    std::vector<std::shared_ptr<data::DSPOptions>>& get_dsp_options();
    std::shared_ptr<data::ModelOption> get_model_options();
    std::shared_ptr<data::Attribute> get_attribute();
    std::shared_ptr<data::ControlOption> get_control_option();
};

using RawModel = Model;
class ModelBuilder {
protected:
    std::shared_ptr<RawModel> raw_model_;

public:
    static ModelBuilder build() {
        return ModelBuilder();
    }

    explicit ModelBuilder() : raw_model_(std::make_shared<RawModel>()) {}
    explicit ModelBuilder(std::shared_ptr<RawModel> raw_model) : raw_model_(raw_model) {}
    ~ModelBuilder();

    data::AttributeBuilder build_attribute() const;
    data::BinaryBuilder build_binary() const;
    data::BufferBuilder build_buffer() const;
    data::ControlOptionBuilder build_control_option() const;
    data::GraphInfoBuilder build_graph_info() const;
    data::ModelOptionBuilder build_model_option() const;
    data::NPUOptionsBuilder build_npu_options() const;
    data::DSPOptionsBuilder build_dsp_options() const;
    data::OperatorBuilder build_operator() const;
    data::OperatorOptionsBuilder build_operator_options() const;
    data::RegionBuilder build_region() const;
    data::ScalarBuilder build_scalar() const;
    data::TensorBuilder build_tensor() const;

    // return RawModel and raw_model_ will be moved and nullptr.
    auto create() {
        return std::move(raw_model_);
    }

    auto get_model() {
        return raw_model_;
    }
};

};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_MODEL_HPP_
