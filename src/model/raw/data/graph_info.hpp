#ifndef SRC_MODEL_RAW_DATA_GRAPH_INFO_HPP_
#define SRC_MODEL_RAW_DATA_GRAPH_INFO_HPP_

#include "model/raw/model.hpp"

namespace enn {
namespace model {
namespace raw {
namespace data {

/*
 * GraphInfo is {name, indexes of input, indexes of output} from SubGraph
 */
class GraphInfo {
private:
    std::string name;
    std::vector<int32_t> inputs;
    std::vector<int32_t> outputs;

    friend class GraphInfoBuilder;

public:
    std::string get_name() {
        return name;
    }

    std::vector<int32_t>& get_inputs() {
        return inputs;
    }

    std::vector<int32_t>& get_outputs() {
        return outputs;
    }
};

class GraphInfoBuilder : public ModelBuilder {
private:
    std::shared_ptr<GraphInfo> graph_info_;

public:
    explicit GraphInfoBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};

    GraphInfoBuilder& add_graph_info() {
        graph_info_ = std::make_unique<GraphInfo>();
        return *this;
    }

    GraphInfoBuilder& get_graph_info(uint32_t index) {
        // Reuqired, ToDo(empire.jung, 6/30): Add exception handling about out of range
        graph_info_ = raw_model_->get_graph_infos().at(index);
        return *this;
    }

    GraphInfoBuilder& set_name(std::string name) {
        graph_info_->name = name;
        return *this;
    }

    GraphInfoBuilder& set_inputs(std::vector<int32_t> inputs) {
        graph_info_->inputs = inputs;
        return *this;
    }

    GraphInfoBuilder& set_outputs(std::vector<int32_t> outputs) {
        graph_info_->outputs = outputs;
        return *this;
    }

    void build() {
        raw_model_->graph_infos_.push_back(std::move(graph_info_));
    }
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_GRAPH_INFO_HPP_
