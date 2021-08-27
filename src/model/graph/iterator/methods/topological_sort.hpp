#ifndef MODEL_GRAPH_ITERATOR_METHODS_TOPOLOGICAL_SORT_HPP_
#define MODEL_GRAPH_ITERATOR_METHODS_TOPOLOGICAL_SORT_HPP_

#include <queue>
#include <map>
#include "model/graph/iterator/iterator.hpp"

namespace enn {
namespace model {
namespace graph {
namespace iterator {

/* Subclasses from Iterator class are concrete methods of iterator */
template <typename Vertex, typename Edge>
class TopologicalSort : public Iterator<Vertex, Edge> {
 public:
    ~TopologicalSort() = default;

    // Iterator<Vertex, Edge> begin() override { return *this; }
    // Iterator<Vertex, Edge> end() override { return *this; }

    TopologicalSort& operator++() override {
        for (const auto& neighbor : this->graph_[this->current_]) {
            if (--in_degree_board[neighbor.first] == 0) {
                q.push(neighbor.first);
            }
        }

        if (q.empty()) {
            this->current_ = nullptr;
        } else {
            this->current_ = q.front();
            q.pop();
        }
        return *this;
    }

 private:
    friend class Graph<Vertex, Edge>;  // Graph only can create this calss.

    TopologicalSort() = default;
    TopologicalSort(Graph<Vertex, Edge>& graph, Vertex& start_vertex)
        : Iterator<Vertex, Edge>{graph, start_vertex} {
        // initialize data structures.
        for (const auto& adj : this->graph_) {
            for (const auto& neighbor : adj.second) {
                in_degree_board[neighbor.first]++;
            }
        }
    }

    std::queue<Vertex> q;
    std::map<Vertex, int> in_degree_board;
};


};  // namespace iterator
};  // namespace graph
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_TRAVERSAL_TOPOLOGICAL_SORT_HPP_
