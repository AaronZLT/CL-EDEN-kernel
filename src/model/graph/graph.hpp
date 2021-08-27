#ifndef SRC_MODEL_GRAPH_GRAPH_HPP_
#define SRC_MODEL_GRAPH_GRAPH_HPP_

#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#include <map>

#include "common/extended_type_traits.hpp"
#include "common/enn_debug.h"

namespace enn {
namespace model {
namespace graph {

namespace iterator {
template <typename V, typename E> class TopologicalSort;
template <typename V, typename E> class DepthFirstSearch;
template <typename V, typename E> class BreadthFirstSearch;
}

// Graph Container can deal with Vertex and Edge as nothing but std::shared_tpr for memory management.
template <typename Vertex, typename Edge,
          typename = typename std::enable_if_t<enn::util::is_shared_ptr<Vertex>::value>,
          typename = typename std::enable_if_t<enn::util::is_shared_ptr<Edge>::value>>
class Graph {
 public:
    using Ptr = std::shared_ptr<Graph>;

 private:
    using Neighbor = std::pair<Vertex, Edge>;
    using Neighbors = std::vector<Neighbor>;
    using AdjacencyList = std::map<Vertex, Neighbors>;

    //  * Key is Vertex(prev vertex).
    //  * Value is vector of Vertex(next vertex) and Edge(edge connecting between vertices).
    //  -----------------------------------------------------
    //  | Vertex -> [ (Vertex, Edge), (Vertex, Edge), ... ] |
    //  | Vertex -> [ (Vertex, Edge), (Vertex, Edge), ... ] |
    //  | Vertex -> [ (Vertex, Edge), (Vertex, Edge), ... ] |
    //  |    ...                                            |
    //  -----------------------------------------------------
    AdjacencyList adjacency_list_;
    Vertex default_start_vertex_;  // a start vertex in graph
    Vertex default_end_vertex_;    // a end vertex in graph

    auto begin() {
        return adjacency_list_.begin();
    }

    auto end() {
        return adjacency_list_.end();
    }

    // iterator classes that can access adjacency_list_ in Graph.
    friend class iterator::TopologicalSort<Vertex, Edge>;
    friend class iterator::DepthFirstSearch<Vertex, Edge>;
    friend class iterator::BreadthFirstSearch<Vertex, Edge>;

 public:
    Graph() = default;
    ~Graph() = default;

    template <template <typename, typename> typename Iterator>
    class Order {
     private:
        Graph<Vertex, Edge>& graph_;
        Vertex& start_vertex_;
     public:
        Order(Graph<Vertex, Edge>& graph, Vertex& start_vertex)
            : graph_{graph}, start_vertex_{start_vertex} {}
        Iterator<Vertex, Edge> begin() { return Iterator<Vertex, Edge>{graph_, start_vertex_}; }
        Iterator<Vertex, Edge> end() { return Iterator<Vertex, Edge>{}; }
    };

    // Factory method for iterator injected with this Graph.
    //   Clients inject a iterater they want to set.
    template <template <typename, typename> typename Iterator>
    auto order() {
        return Order<Iterator>(*this, default_start_vertex_);
    }

    // Clients can also give start vertex.
    template <template <typename, typename> typename Iterator>
    auto order(Vertex start_vertex) {
        return Order<Iterator>(*this, start_vertex);
    }

    // return neighbor list of vertex given
    const Neighbors& operator[](const Vertex& vertex) const {
        return adjacency_list_.at(vertex);
    }

    Neighbors& operator[](const Vertex& vertex) {
        return adjacency_list_.at(vertex);
    }

    template <typename V>
    Graph& add_vertex(V&& vertex) {
        adjacency_list_[std::forward<V>(vertex)];
        return *this;
    }

    template <typename V1, typename E, typename V2>
    Graph& add_neighbor(V1&& from_vertex, E&& edge, V2&& to_vertex) {
        ENN_DBG_COUT << "-From op/op_list["      << (int)from_vertex->get_id()
                     << "] -> Tensor["   << (int)edge->get_id()
                     << "] -> To op/op_list[" << (int)to_vertex->get_id()
                     << "] Coneected"    << std::endl;
        adjacency_list_[std::forward<V1>(from_vertex)]
            .push_back({std::forward<V2>(to_vertex), std::forward<E>(edge)});
        return *this;
    }

    template <typename V>
    Graph& set_start_vertex(V&& vertex) {
        default_start_vertex_ = std::forward<V>(vertex);
        return *this;
    }

    template <typename V>
    Graph& set_end_vertex(V&& vertex) {
        default_end_vertex_ = std::forward<V>(vertex);
        return *this;
    }

    auto get_start_vertex() const {
        return default_start_vertex_;
    }

    auto get_end_vertex() const {
        return default_end_vertex_;
    }

    uint32_t vertex_count() const {
        return adjacency_list_.size();
    }

    // TODO(yc18.cho): optimize for performance after inspecting how it will be used in client code.
    uint32_t edge_count() const {
        auto edge_count = 0;
        for (const auto& it : adjacency_list_) {
            edge_count += it.second.size();
        }
        return edge_count;
    }
};


};  // namespace graph
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_GRAPH_GRAPH_HPP_