#include "model/pool/manager.hpp"

#include <stdexcept>
#include <type_traits>

#include "model/model.hpp"
#include "common/enn_debug.h"

#include "medium/enn_medium_utils.hpp"
namespace enn {
namespace model {
namespace pool {

template <typename T>
void Manager::add_model(std::shared_ptr<T> model) {}

template <>
void Manager::add_model(std::shared_ptr<PrototypeModel> model) {
    // TODO(yc18.cho): raise user defined exception and handle exception in client code, not here.
    check_model_id(model);
    prototype_model_pool_.insert(model->get_id(), model);
    std::unique_lock<std::mutex> lock(mutex_);
    pid_to_model_id_[enn::util::get_caller_pid()].push_back(model->get_id());
}

template <>
void Manager::add_model(ExecutableModel::Ptr model) {
    // TODO(yc18.cho): raise user defined exception and handle exception in client code, not here.
    check_model_id(model);
    executable_model_pool_.insert(model->get_id(), model);
    std::unique_lock<std::mutex> lock(mutex_);
    proto_to_exec_map_[model->get_model_id()].insert(model->get_id());
}

int32_t Manager::add_model_mem_objects(PrototypeModel::ID id, Manager::MemObject memory_obj) {
    check_model_id<PrototypeModel>(id);  // check model ID is valid
    model_mem_object_pool[id].push_back(memory_obj);
    return memory_obj->fd;
}

Manager::IteratorPair Manager::get_model_mem_object_pool(PrototypeModel::ID id) {
    check_model_id<PrototypeModel>(id);  // check model ID is valid
    return std::make_pair(model_mem_object_pool[id].begin(), model_mem_object_pool[id].end());
}

template <typename T>
std::shared_ptr<T> Manager::get_model(typename T::ID id) const {}

template <>
std::shared_ptr<PrototypeModel> Manager::get_model(PrototypeModel::ID id) const {
    // TODO(yc18.cho): raise user defined exception and handle exception in client code, not here.
    return prototype_model_pool_.find(id);
}

template <>
ExecutableModel::Ptr Manager::get_model(ExecutableModel::ID id) const {
    // TODO(yc18.cho): raise user defined exception and handle exception in client code, not here.
    return executable_model_pool_.find(id);
}

template <typename T>
void Manager::release_model(typename T::ID id) {}

template <>
void Manager::release_model<PrototypeModel>(PrototypeModel::ID id) {
    // TODO(yc18.cho): raise user defined exception and handle exception in client code, not here.
    auto prototype_model = prototype_model_pool_.find(id);
    prototype_model_pool_.remove(id);
    model_mem_object_pool.erase(id);

    // remove ExecutableModels from this PrototypeModel.
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto executable_model_id : proto_to_exec_map_[id])
        executable_model_pool_.remove(executable_model_id);
    proto_to_exec_map_.erase(id);

    // "id >> 48" is process id of service.
    // Cannot use util::get_caller_pid() because,
    // when close model is called by timeout or ANR,
    // get_caller_pid() will return Monitor's pid,
    // Not service's pid, which is used in Open Model.
    auto& vec_model_id = pid_to_model_id_[id >> 48];
    vec_model_id.erase(std::remove(vec_model_id.begin(), vec_model_id.end(), id), vec_model_id.end());
}

template <>
void Manager::release_model<ExecutableModel>(ExecutableModel::ID id) {
    // TODO(yc18.cho): raise user defined exception and handle exception in client code, not here.
    auto proto_model_id = executable_model_pool_.find(id)->get_model_id();
    executable_model_pool_.remove(id);
    std::unique_lock<std::mutex> lock(mutex_);
    proto_to_exec_map_[proto_model_id].erase(id);
}

template <typename T>
size_t Manager::count() const { return 0;}

template <>
size_t Manager::count<PrototypeModel>() const {
    return prototype_model_pool_.size();
}

template <>
size_t Manager::count<ExecutableModel>() const {
    return executable_model_pool_.size();
}

template <typename T>
void Manager::check_model_id(std::shared_ptr<T> model) const {
    if (model->get_id() == T::DefaultID) {
#ifndef VELOCE_SOC
        throw std::invalid_argument(
            "[invalid argument] check_model_id: The id in PrototypeModel object is invalid.");
#endif
    }
}

template <typename T>
void Manager::check_model_id(typename T::ID id) const {
    if (id == T::DefaultID) {
#ifndef VELOCE_SOC
        throw std::invalid_argument(
            "[invalid argument] check_model_id: The id in PrototypeModel object is invalid.");
#endif
    }
}

std::vector<PrototypeModel::ID> Manager::get_model_id_list(int caller_pid) {
    return pid_to_model_id_[caller_pid];
}



};  // namespace pool
};  // namespace model
};  // namespace enn
