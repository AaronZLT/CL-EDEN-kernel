#ifndef SRC_MODEL_PARSER_STRATEGY_CGO_PARSE_STRATEGY_HPP_
#define SRC_MODEL_PARSER_STRATEGY_CGO_PARSE_STRATEGY_HPP_

#include <mutex>
#include <unordered_map>
#include "model/schema/schema_cgo.h"
#include "model/parser/strategy/strategy.hpp"
#include "model/raw/model.hpp"

namespace enn {
namespace model {

using namespace ofi::rawgraph;
using namespace enn::model::raw::data;

class CgoParseStrategy : public ParseStrategy {
public:
    explicit CgoParseStrategy(const std::shared_ptr<ModelMemInfo> model,
                              const std::shared_ptr<std::vector<std::shared_ptr<ModelMemInfo>>> params);

    void pre_execute() override;
    void parse_operators() override;
    void parse_tensors() override;
    void parse_attribute() override;
    void post_execute() override;
    std::shared_ptr<raw::Model> result() override;
    void print() override;

private:
    const ofi::rawgraph::fb_OfiRawGraph* ofi_raw_graph;
    raw::ModelBuilder model_builder = raw::ModelBuilder::build();

    std::shared_ptr<std::vector<std::shared_ptr<ModelMemInfo>>> cgo_params;

    std::mutex mutex_tensor_name_map;
    std::unordered_map<uint32_t, std::string> new_tensor_name_map;
    void add_new_tensor_name(uint32_t index, std::string name);

    void parse_operators_impl(const ofi::rawgraph::fb_OfiMacroSubGraph* msg);
    void parse_dsp_binary_impl(std::vector<std::string>& lib_names, std::vector<int32_t>& binary_indexes);
    void parse_graph_infos_impl();

    const flatbuffers::Vector<flatbuffers::Offset<fb_OfiBuffer>>* core_buffers = nullptr;
    const flatbuffers::Vector<flatbuffers::Offset<fb_OfiScalar>>* core_scalars = nullptr;
    void generate_buffer_indexes(const flatbuffers::Vector<uint32_t>* in_container, bool is_scalar,
                                 std::vector<int32_t>* out_container);

    // TODO: Disable in release version
    bool validate_operators();
    bool validate_tensors();
    bool validate_attribute();
};

};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_PARSER_STRATEGY_CGO_PARSE_STRATEGY_HPP_
