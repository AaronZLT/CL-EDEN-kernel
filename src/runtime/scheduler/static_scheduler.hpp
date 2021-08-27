#ifndef RUNTIME_SCHEDULER_STATIC_SCHEDULER_HPP_
#define RUNTIME_SCHEDULER_STATIC_SCHEDULER_HPP_

#include <map>
#include <vector>
#include <string>
#include <memory>

#include "model/model.hpp"

#include "model/graph_types.hpp"
#include "model/component/operator/operator.hpp"
#include "model/component/operator/operator_list_builder.hpp"
#include "model/component/tensor/feature_map_builder.hpp"
#include "model/graph/iterator/methods/topological_sort.hpp"

namespace enn {
namespace runtime {
namespace schedule {

enum class ModelTrait {
    Default
};

using namespace enn::model::component;

class IStaticSchedule {
 public:
    virtual ~IStaticSchedule() = default;
    // Group Operators into OperatorList
    virtual void arrange(model::Model::Ptr target_model) = 0;
};

class DefaultStaticSchedule : public IStaticSchedule {
    void arrange(model::Model::Ptr target_model) override {
        ENN_DBG_COUT << "Schedule origin graph to op_list graph" << std::endl;
        model::ScheduledGraph::Ptr scheduled_graph = std::make_shared<model::ScheduledGraph>();

        model::Accelerator target_device = model::Accelerator::SIZE;
        std::vector<OperatorList::Ptr> operator_list_vector;

        OperatorListBuilder operator_list_builder;
        // loop origin graph to collect same taget device operator.
        for (auto& opr_ptr : target_model->get_origin_graph()->order<enn::model::graph::iterator::TopologicalSort>()) {
            if (static_cast<int>(opr_ptr->get_id()) < 0) {
                // Skip the Virtual Input & Virtual Output.
                continue;
            }

            if (target_device == model::Accelerator::SIZE) {
                // for first operator, add operator to list and update target device
                operator_list_builder.build(target_model->get_id())
                                     .set_attribute(target_model->get_attribute());
                operator_list_builder.add_operator(opr_ptr);
                target_device = opr_ptr->get_accelerator();
            } else if (target_device == opr_ptr->get_accelerator()) {
                // for same target device operator, add operator to operator list.
                operator_list_builder.add_operator(opr_ptr);
            } else {
                // when visit different target device operator,
                // get the operator list made up to now.
                auto op_list = operator_list_builder.set_accelerator(target_device)
                                                    .create();

                OperatorList::Ptr prev_op_list_vertex;
                if (op_list->get_size() != 0) {
                    operator_list_vector.push_back(op_list);
                    prev_op_list_vertex = std::move(op_list);
                    op_list = operator_list_builder
                                .build(target_model->get_id())
                                .set_attribute(target_model->get_attribute())
                                .create();

                    FeatureMapBuilder feature_map_builder;
                    FeatureMap::Ptr fm = feature_map_builder.set_id(-1).create();
                    scheduled_graph->add_neighbor(prev_op_list_vertex, fm, op_list);
                }

                target_device = opr_ptr->get_accelerator();
                operator_list_builder = OperatorListBuilder(op_list);
                operator_list_builder.add_operator(opr_ptr);
            }
        }

        // create last operator_list.
        if (operator_list_builder.get()->get_size() != 0) {
            operator_list_builder.set_accelerator(target_device);
            operator_list_vector.push_back(operator_list_builder.create());
        }

        // configure scheduled_graph to set in the enn model.
        scheduled_graph->set_start_vertex(operator_list_vector.at(0));
        scheduled_graph->set_end_vertex(operator_list_vector.at(operator_list_vector.size() - 1));
        for (auto& v : operator_list_vector) {
            scheduled_graph->add_vertex(v);
        }

        target_model->set_scheduled_graph(scheduled_graph);
        ENN_DBG_COUT << "Schedule origin graph to op_list graph Completed" << std::endl;
    }
};

// When a new scheduling method is required, create a class extending IStaticScheule.
//  And when the model that needs that scheduling comes in, set an instance of that class to the schedule_.
class StaticScheduler {
 public:
    StaticScheduler() : preset_id(0), pref_mode(0), target_latency(0), tile_num(1), core_affinity(0), priority(0) {}
    StaticScheduler& set_model(model::Model::Ptr model) {
        target_model_ = model;
        auto model_trait = analyze_model();
        switch (model_trait) {
            case ModelTrait::Default:
                schedule_ = std::make_unique<DefaultStaticSchedule>();
                break;
        }
        return *this;
    }

    StaticScheduler& set_preset_id(uint32_t preset_id_from_client) {
        preset_id = preset_id_from_client;
        return *this;
    }

    StaticScheduler& set_pref_mode(uint32_t pref_mode_from_client) {
        pref_mode = pref_mode_from_client;
        return *this;
    }

    StaticScheduler& set_target_latency(uint32_t target_latency_from_client) {
        target_latency = target_latency_from_client;
        return *this;
    }

    StaticScheduler& set_tile_num(uint32_t tile_num_from_client) {
        tile_num = tile_num_from_client;
        return *this;
    }

    StaticScheduler& set_core_affinity(uint32_t core_affinity_form_client) {
        core_affinity = core_affinity_form_client;
        return *this;
    }

    StaticScheduler& set_priority(uint32_t priority_form_client) {
        priority = priority_form_client;
        return *this;
    }

    void run() {
        schedule_->arrange(target_model_);
        set_preferences();
    }

 private:
    ModelTrait analyze_model() {
        // Analyze the target_model_ and then return ModelTrait
        return ModelTrait::Default;
    }

    void set_preferences() {
        using namespace enn::model::graph::iterator;
        for (auto& op_list : target_model_->get_scheduled_graph()->order<BreadthFirstSearch>()) {
            auto op_list_builder = OperatorListBuilder(op_list);
            op_list_builder.set_preset_id(preset_id)
                           .set_pref_mode(pref_mode)
                           .set_target_latency(target_latency)
                           .set_tile_num(tile_num)
                           .set_core_affinity(core_affinity)
                           .set_priority(priority);
        }
    }

    model::Model::Ptr target_model_;
    std::unique_ptr<IStaticSchedule> schedule_;

    // Preferences From Client
    uint32_t preset_id;
    uint32_t pref_mode;         // for EDEN backward compatibility
    uint32_t target_latency;    // for DVFS hint
    uint32_t tile_num;          // for batch processing hint
    uint32_t core_affinity;
    uint32_t priority;
};

};  // namespace schedule
};  // namespace runtime
};  // namespace enn

#endif  // RUNTIME_SCHEDULER_STATIC_SCHEDULER_HPP_
