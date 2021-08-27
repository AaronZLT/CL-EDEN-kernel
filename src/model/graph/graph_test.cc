#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>


#include "graph.hpp"

#include "model/component/operator/operator_builder.hpp"
#include "model/component/operator/operator_list.hpp"
#include "model/component/tensor/feature_map_builder.hpp"
#include "model/graph/iterator/methods/topological_sort.hpp"
#include "model/graph/iterator/methods/depth_first_search.hpp"
#include "model/graph/iterator/methods/breadth_first_search.hpp"
#include "model/graph/iterator/methods/linear_search.hpp"

using namespace enn::model::component;
using namespace enn::model::graph;

class GraphTest : public testing::Test {
 protected:
    void create_components(int op_count, int fm_count) {
        create_operators(op_count);
        create_feature_maps(fm_count);
    }

    void create_operators(int op_count) {
        // create operators with id and name that are i until operator_count
        OperatorBuilder operator_builder;
        for (int i = 0; i < op_count; i++) {
            operators.push_back(operator_builder.set_id(i)
                                                .set_name(std::to_string(i))
                                                .create());
        }
    }

    void create_feature_maps(int fm_count) {
        FeatureMapBuilder feature_map_builder;
        // create feature_map with id and name that are i until feature_map
        for (int i = 0; i < fm_count; i++) {
            feature_maps.push_back(feature_map_builder.set_id(i)
                                                      .set_name(std::to_string(i))
                                                      .create());
        }
    }

    void set_adjacency(int prev_vertex, int edge, int next_vertex) {
        OperatorBuilder prev_op_builder(operators[prev_vertex]);
        prev_op_builder.add_out_tensor(feature_maps[edge]);
        OperatorBuilder next_op_builder(operators[prev_vertex]);
        next_op_builder.add_in_tensor(feature_maps[edge]);
        FeatureMapBuilder edge_builder(feature_maps[edge]);
        edge_builder.set_prev_operator(operators[prev_vertex]);
        edge_builder.add_next_operator(operators[next_vertex]);
    }

    auto create_graph() {
        Graph<Operator::Ptr, FeatureMap::Ptr> graph;
        // create adjacency list in graph
        for (auto& opr : operators) {
            graph.add_vertex(opr);
        }
        return graph;
    }
    std::vector<Operator::Ptr> operators;
    std::vector<FeatureMap::Ptr> feature_maps;

 public:
    void graph_tc_1_in_1_out(Graph<Operator::Ptr, FeatureMap::Ptr>& graph) {
        // Graph for this case is like below:
        //                      O --operators[0]        // Virtual Input
        //                      | --feature_maps[0]     // model - input buffer
        //                      O --operators[1]        // model - operator
        //  feature_maps[2] -- / \ --feature_maps[1]    // model - operand
        //     operators[3]-- O   O --operators[2]      // model - operator
        //  feature_maps[4] -- \ / -- feature_maps[3]   // model - operand
        //       operators[4]-- O --operators[4]        // model - operator
        //                      | --feature_maps[5]     // model - output buffer
        //                      O --operators[5]        // Virtual Output
        graph.add_neighbor(operators[0], feature_maps[0], operators[1])
             .add_neighbor(operators[1], feature_maps[2], operators[3])
             .add_neighbor(operators[1], feature_maps[1], operators[2])
             .add_neighbor(operators[3], feature_maps[4], operators[4])
             .add_neighbor(operators[2], feature_maps[3], operators[4])
             .add_neighbor(operators[4], feature_maps[5], operators[5])
             .set_start_vertex(operators[0]);  // set start vertex of this graph
    }
    void graph_tc_2_in_1_out(Graph<Operator::Ptr, FeatureMap::Ptr>& graph) {
        // Graph for this case is like below:
        //       [1] - [3] - [5] - [v_out]
        //      /            /
        // [v_in]           /
        //      \          /
        //       [2] - [4]
        const int v_in = 0;
        const int v_out = 6;
        graph.add_neighbor(operators[v_in], feature_maps[0], operators[1])
             .add_neighbor(operators[v_in], feature_maps[1], operators[2])
             .add_neighbor(operators[1], feature_maps[2], operators[3])
             .add_neighbor(operators[2], feature_maps[3], operators[4])
             .add_neighbor(operators[3], feature_maps[4], operators[5])
             .add_neighbor(operators[4], feature_maps[5], operators[5])
             .add_neighbor(operators[5], feature_maps[6], operators[v_out])
             .set_start_vertex(operators[v_in]);  // set start vertex of this graph
    }
    void graph_tc_1_in_2out(Graph<Operator::Ptr, FeatureMap::Ptr>& graph) {
        // Graph for this case is like below:
        //              [2] - [3]           -- [5]
        //            /          \        /     |
        // [v_in] - [1] - [7] - [8] - [4]      [v_out]
        //            \          /      \     /
        //              [9] - [10]        [6]
        const int v_in = 0;
        const int v_out = 11;
        graph.add_neighbor(operators[v_in], feature_maps[0], operators[1])
             .add_neighbor(operators[1], feature_maps[1], operators[2])
             .add_neighbor(operators[1], feature_maps[2], operators[7])
             .add_neighbor(operators[1], feature_maps[3], operators[9])
             .add_neighbor(operators[2], feature_maps[4], operators[3])
             .add_neighbor(operators[7], feature_maps[5], operators[8])
             .add_neighbor(operators[9], feature_maps[6], operators[10])
             .add_neighbor(operators[3], feature_maps[7], operators[8])
             .add_neighbor(operators[10], feature_maps[8], operators[8])
             .add_neighbor(operators[8], feature_maps[9], operators[4])
             .add_neighbor(operators[4], feature_maps[10], operators[5])
             .add_neighbor(operators[4], feature_maps[11], operators[6])
             .add_neighbor(operators[5], feature_maps[12], operators[v_out])
             .add_neighbor(operators[6], feature_maps[13], operators[v_out])
             .set_start_vertex(operators[v_in]);  // set start vertex of this graph
    }
};


TEST_F(GraphTest, verify_adjacencies_between_opr_and_fm) {
    create_components(6, 6);
    // Set adjacency between vertices and edge
    set_adjacency(0, 0, 1);
    set_adjacency(1, 1, 2);
    set_adjacency(1, 2, 3);
    set_adjacency(2, 3, 4);
    set_adjacency(3, 4, 4);
    set_adjacency(4, 5, 5);
    ASSERT_EQ(operators[0]->out_tensors[0]->get_id(), 0);
    ASSERT_EQ(operators[1]->out_tensors[0]->get_id(), 1);
    ASSERT_EQ(operators[1]->out_tensors[1]->get_id(), 2);
    ASSERT_EQ(operators[2]->out_tensors[0]->get_id(), 3);
    ASSERT_EQ(operators[3]->out_tensors[0]->get_id(), 4);
    ASSERT_EQ(operators[4]->out_tensors[0]->get_id(), 5);
}

TEST_F(GraphTest, test_checking_capacity_of_graph_created) {
    create_components(6, 6);
    auto graph = create_graph();
    graph_tc_1_in_1_out(graph);

    // check sanity and capacity of graph
    EXPECT_EQ(graph.get_start_vertex(), operators[0]);
    EXPECT_EQ(graph.vertex_count(), 6);
    EXPECT_EQ(graph.edge_count(), 6);
}

TEST_F(GraphTest, check_traverse_graph_with_Topological_1_in_1_out_basic_for_loop) {
    create_components(6, 6);
    auto graph = create_graph();
    graph_tc_1_in_1_out(graph);

    int expected_visit_sequence[] = { 0, 1, 2, 3, 4, 5};
    for (auto it = graph.order<enn::model::graph::iterator::TopologicalSort>().begin();
         it != graph.order<enn::model::graph::iterator::TopologicalSort>().end(); ++it) {
            EXPECT_EQ((*it)->get_id(), expected_visit_sequence[(*it)->get_id()]);
    }
}

// -------------------------------TOPOLOGICAL SORT ---------------------------- //
TEST_F(GraphTest, check_traverse_graph_with_Topological_1_in_1_out_range_based) {
    create_components(6, 6);
    auto graph = create_graph();
    graph_tc_1_in_1_out(graph);

    int expected_visit_sequence[] = { 0, 1, 2, 3, 4, 5};
    for (auto& opr : graph.order<enn::model::graph::iterator::TopologicalSort>()) {
        EXPECT_EQ(opr->get_id(), expected_visit_sequence[opr->get_id()]);
    }
}

TEST_F(GraphTest, check_traverse_graph_with_topological_2_in_1_out_range_based) {
    create_components(7, 7);
    auto graph = create_graph();
    graph_tc_2_in_1_out(graph);

    int expected_visit_sequence[] = { 0, 1, 2, 3, 4, 5, 6};
    int expected_count = 0;
    for (auto& opr : graph.order<enn::model::graph::iterator::TopologicalSort>()) {
        EXPECT_EQ(opr->get_id(), expected_visit_sequence[expected_count++]);
    }
}

TEST_F(GraphTest, check_traverse_graph_with_topological_1_in_2_out_range_based) {
    create_components(12, 14);
    auto graph = create_graph();
    graph_tc_1_in_2out(graph);

    int expected_visit_sequence[] = { 0, 1, 2, 7, 9, 3, 10, 8, 4, 5, 6, 11};
    int expected_count = 0;
    for (auto& opr : graph.order<enn::model::graph::iterator::TopologicalSort>()) {
        EXPECT_EQ(opr->get_id(), expected_visit_sequence[expected_count++]);
    }
}

// ------------------------------- DFS -------------------------------- //
TEST_F(GraphTest, check_traverse_graph_with_DFS_1_in_1_out_range_based) {
    create_components(6, 6);
    auto graph = create_graph();
    graph_tc_1_in_1_out(graph);

    int expected_visit_sequence[] = { 0, 1, 2, 4, 5, 3};
    int expected_count = 0;
    for (auto& opr : graph.order<enn::model::graph::iterator::DepthFirstSearch>()) {
        EXPECT_EQ(opr->get_id(), expected_visit_sequence[expected_count++]);
    }
}

TEST_F(GraphTest, check_traverse_graph_with_DFS_2_in_1_out_range_based) {
    create_components(7, 7);
    auto graph = create_graph();
    graph_tc_2_in_1_out(graph);

    int expected_visit_sequence[] = { 0, 2, 4, 5, 6, 1, 3};
    int expected_count = 0;
    for (auto& opr : graph.order<enn::model::graph::iterator::DepthFirstSearch>()) {
        EXPECT_EQ(opr->get_id(), expected_visit_sequence[expected_count++]);
    }
}

TEST_F(GraphTest, check_traverse_graph_with_DFS_1_in_2_out_range_based) {
    create_components(12, 14);
    auto graph = create_graph();
    graph_tc_1_in_2out(graph);

    int expected_visit_sequence[] = { 0, 1, 9, 10, 8, 4, 6, 11, 5, 7, 2, 3};
    int expected_count = 0;
    for (auto& opr : graph.order<enn::model::graph::iterator::DepthFirstSearch>()) {
        EXPECT_EQ(opr->get_id(), expected_visit_sequence[expected_count++]);
    }
}

// ------------------------------- BFS -------------------------------- //
TEST_F(GraphTest, check_traverse_graph_with_BFS_1_in_1_out_range_based) {
    create_components(6, 6);
    auto graph = create_graph();
    graph_tc_1_in_1_out(graph);

    int expected_visit_sequence[] = { 0, 1, 3, 2, 4, 5};
    int expected_count = 0;
    for (auto& opr : graph.order<enn::model::graph::iterator::BreadthFirstSearch>()) {
        EXPECT_EQ(opr->get_id(), expected_visit_sequence[expected_count++]);
    }
}

TEST_F(GraphTest, check_traverse_graph_with_BFS_2_in_1_out_range_based) {
    create_components(7, 7);
    auto graph = create_graph();
    graph_tc_2_in_1_out(graph);

    int expected_visit_sequence[] = { 0, 1, 2, 3, 4, 5, 6};
    int expected_count = 0;
    for (auto& opr : graph.order<enn::model::graph::iterator::BreadthFirstSearch>()) {
        EXPECT_EQ(opr->get_id(), expected_visit_sequence[expected_count++]);
    }
}

TEST_F(GraphTest, check_traverse_graph_with_BFS_1_in_2_out_range_based) {
    create_components(12, 14);
    auto graph = create_graph();
    graph_tc_1_in_2out(graph);

    int expected_visit_sequence[] = { 0, 1, 2, 7, 9, 3, 8, 10, 4, 5, 6, 11};
    int expected_count = 0;
    for (auto& opr : graph.order<enn::model::graph::iterator::BreadthFirstSearch>()) {
        EXPECT_EQ(opr->get_id(), expected_visit_sequence[expected_count++]);
    }
}

// ----------------------------- Linear Search ------------------------------ //
TEST_F(GraphTest, check_traverse_graph_with_Linear_Search_1_in_1_out_range_based) {
    create_components(12, 14);
    auto graph = create_graph();
    graph_tc_1_in_1_out(graph);

    int expected_visit_sequence[] = {0, 1, 3, 4, 5};
    int expected_count = 0;
    for (auto& opr : graph.order<enn::model::graph::iterator::LinearSearch>()) {
        EXPECT_EQ(opr->get_id(), expected_visit_sequence[expected_count++]);
    }
}

TEST_F(GraphTest, check_traverse_graph_with_Linear_Search_2_in_1_out_range_based) {
    create_components(7, 7);
    auto graph = create_graph();
    graph_tc_2_in_1_out(graph);

    int expected_visit_sequence[] = {0, 1, 3, 5, 6};
    int expected_count = 0;
    for (auto& opr : graph.order<enn::model::graph::iterator::LinearSearch>()) {
        EXPECT_EQ(opr->get_id(), expected_visit_sequence[expected_count++]);
    }
}

TEST_F(GraphTest, check_traverse_graph_with_Linear_Search_1_in_2_out_range_based) {
    create_components(12, 14);
    auto graph = create_graph();
    graph_tc_1_in_2out(graph);

    int expected_visit_sequence[] = {0, 1, 2, 3, 8, 4, 5, 11};
    int expected_count = 0;
    for (auto& opr : graph.order<enn::model::graph::iterator::LinearSearch>()) {
        EXPECT_EQ(opr->get_id(), expected_visit_sequence[expected_count++]);
    }
}
