#ifndef MODEL_GRAPH_ITERATOR_DEPTH_FIRST_SEARCH_HPP_
#define MODEL_GRAPH_ITERATOR_DEPTH_FIRST_SEARCH_HPP_

#include <stack>
#include <map>
#include "model/graph/iterator/iterator.hpp"

namespace enn {
namespace model {
namespace graph {
namespace iterator {

template <typename Vertex, typename Edge>
class DepthFirstSearch : public Iterator<Vertex, Edge> {
 public:
    ~DepthFirstSearch() = default;

    DepthFirstSearch& operator++() override {
        for (const auto& neighbor : this->graph_[this->current_]) {
            if (visit_board.find(neighbor.first) == visit_board.end()) {
                stack.push(neighbor.first);
                visit_board[neighbor.first] = false;
            }
        }

        if (stack.empty()) {
            this->current_ = nullptr;
        } else {
            this->current_ = stack.top();
            stack.pop();
            visit_board[this->current_] = true;
        }
        return *this;
    }

 private:
    friend class Graph<Vertex, Edge>;  // Graph only can create this class.

    DepthFirstSearch() = default;
    DepthFirstSearch(Graph<Vertex, Edge>& graph, Vertex& start_vertex)
        : Iterator<Vertex, Edge>{graph, start_vertex} {
        visit_board[this->current_] = true;
    }

    std::stack<Vertex> stack;
    std::map<Vertex, bool> visit_board;
};


};  // namespace iterator
};  // namespace graph
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_TRAVERSAL_DEPTH_FIRST_SEARCH_HPP_