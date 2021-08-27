#ifndef SRC_MODEL_GRAPH_TYPES_HPP_
#define SRC_MODEL_GRAPH_TYPES_HPP_

#include <memory>

#include "model/graph/graph.hpp"
#include "model/component/operator/operator.hpp"
#include "model/component/operator/operator_list.hpp"
#include "model/component/tensor/feature_map.hpp"

namespace enn {
namespace model {

// List of alias for Graph
template <typename Vertex, typename Edge>
using Graph          = enn::model::graph::Graph<Vertex, Edge>;

using OriginalGraph  = Graph<component::Operator::Ptr, component::FeatureMap::Ptr>;
using ScheduledGraph = Graph<component::OperatorList::Ptr, component::FeatureMap::Ptr>;

};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_GRAPH_TYPES_HPP_
