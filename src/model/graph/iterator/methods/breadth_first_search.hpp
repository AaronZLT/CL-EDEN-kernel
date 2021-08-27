#ifndef MODEL_GRAPH_ITERATOR_METHODS_BREADTH_FIRST_SEARCH_HPP_
#define MODEL_GRAPH_ITERATOR_METHODS_BREADTH_FIRST_SEARCH_HPP_

#include <queue>
#include <vector>
#include <map>
#include "model/graph/iterator/iterator.hpp"

namespace enn {
namespace model {
namespace graph {
namespace iterator {

template <typename Vertex, typename Edge>
class BreadthFirstSearch : public Iterator<Vertex, Edge> {
 public:
    Iterator<Vertex, Edge>& operator++() override {
        for (const auto& neighbor : this->graph_[this->current_]) {
            if (visit_board.find(neighbor.first) == visit_board.end()) {
                q.push(neighbor.first);
                visit_board[neighbor.first] = false;
            }
        }

        if (q.empty()) {
            this->current_ = nullptr;
        } else {
            this->current_ = q.front();
            q.pop();
            visit_board[this->current_] = true;
        }
        return *this;
    }

 private:
    friend class Graph<Vertex, Edge>;  // Grpah only can create this class.

    BreadthFirstSearch() = default;
    BreadthFirstSearch(Graph<Vertex, Edge>& graph, Vertex& start_vertex)
        : Iterator<Vertex, Edge>{graph, start_vertex} {
        visit_board[this->current_] = true;
    }
    std::queue<Vertex> q;
    std::map<Vertex, bool> visit_board;
};


};  // namespace iterator
};  // namespace graph
};  // namespace model
};  // namespace enn

#endif  // MODEL_TRAVERSAL_BREADTH_FIRST_SEARCH_HPP_
