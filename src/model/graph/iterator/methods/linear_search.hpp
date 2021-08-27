#ifndef MODEL_GRAPH_ITERATOR_METHODS_LINEAR_SEARCH_HPP_
#define MODEL_GRAPH_ITERATOR_METHODS_LINEAR_SEARCH_HPP_

#include <queue>
#include <map>
#include "model/graph/iterator/iterator.hpp"

namespace enn {
namespace model {
namespace graph {
namespace iterator {


// The LinearSearch Iterator can be applied only to a linear graph, that is,
//  a graph in which all vertices have single incoming edge and single outgoing edge.
// It is for higher performance as this iterator doesn't additional work for traversing algorithm.
template <typename Vertex, typename Edge>
class LinearSearch : public Iterator<Vertex, Edge> {
 public:
    ~LinearSearch() = default;

    LinearSearch& operator++() override {
        if (!this->graph_[this->current_].empty()) this->current_ = this->graph_[this->current_].front().first;
        else this->current_ = nullptr;
        return *this;
    }

 private:
    friend class Graph<Vertex, Edge>;

    LinearSearch() = default;
    LinearSearch(Graph<Vertex, Edge>& graph, Vertex& start_vertex)
        : Iterator<Vertex, Edge>{graph, start_vertex} {
            // nothing to initialize any data structure
        }
};


};  // namespace iterator
};  // namespace graph
};  // namespace model
};  // namespace enn

#endif  // MODEL_GRAPH_ITERATOR_METHODS_LINEAR_SEARCH_HPP_
