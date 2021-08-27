#ifndef MODEL_GRAPH_ITERATOR_ITERATOR_HPP_
#define MODEL_GRAPH_ITERATOR_ITERATOR_HPP_

#include "model/graph/graph.hpp"

namespace enn {
namespace model {
namespace graph {
namespace iterator {


template <typename Vertex, typename Edge,
          typename = typename std::enable_if_t<enn::util::is_shared_ptr<Vertex>::value>,
          typename = typename std::enable_if_t<enn::util::is_shared_ptr<Vertex>::value>>
class Iterator {

 public:
    virtual ~Iterator() = default;

    // Deference to vertex
    Vertex& operator*() {
        return current_;
    }

    // Main function to traverse graph.
    //  All subclases should override this function for the logic of specfic iterator.
    //  It should keep object iterating once in current_.
    virtual Iterator& operator++() = 0;

    // overload operator != to be used in for statement
    bool operator!=(const Iterator& other) {
        return current_ != other.current_;
    }

 protected:
    Iterator()
        : current_{nullptr} {}

    Iterator(Graph<Vertex, Edge>& graph, Vertex start_vertex)
        : graph_{graph}, current_{start_vertex} {}

    Graph<Vertex, Edge> graph_;
    Vertex current_;
};


};  // namespace iterator
};  // namespace graph
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_GRAPH_ITERATOR_ITERATOR_HPP_
