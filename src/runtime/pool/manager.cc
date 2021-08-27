#include "runtime/pool/manager.hpp"

#include <stdexcept>

#include "model/model.hpp"
#include "common/enn_debug.h"

namespace enn {
namespace runtime {
namespace pool {

// Calculate maximum values of Model and ExecutableModel for instantiating the Identifier.
using PUID = ClientProcess::UniqueID;
using MUID = Model::UniqueID;
using EMUID = ExecutableModel::UniqueID;
constexpr FullIDType ModelIDMax
    = (PUID::Max << (PUID::Offset - MUID::Offset)) | MUID::Max;
constexpr FullIDType ExecutableModelIDMax
    = (MUID::Max << (MUID::Offset - EMUID::Offset)) | EMUID::Max;

// Define ID's type by setting the configuration of ID as a template argument of Identifier class.
using ClientProcessID = Identifier<FullIDType, PUID::Max, PUID::Offset>;
using ModelID = Identifier<FullIDType, ModelIDMax, MUID::Offset>;
using ExecutableModelID = Identifier<FullIDType, ExecutableModelIDMax, EMUID::Offset>;
using ExecuteRequestID = ExecutableModelID;  // ExecuteRequest's ID is reused as ExecutableModel's ID

template <>
void Manager::add(const ClientProcess::Ptr& client_process) {
    if (!tree_pool_.add<underlying_cast(TreePoolConfig::CLIENT_PROCESS_LEVEL)>(client_process))
        throw std::runtime_error(
                "[Error] Fail to add a ClientProecss(ID: 0x" + client_process->get_id().to_string());
    ENN_DBG_COUT << client_process->to_string() << " is added to the Pool" << std::endl;
}

template <>
void Manager::add(const Model::Ptr& model) {
    if (!tree_pool_.add<underlying_cast(TreePoolConfig::MODEL_LEVEL)>(model))
        throw std::runtime_error(
                "[Error] Fail to add a Model(ID: 0x" + model->get_id().to_string());
    ENN_DBG_COUT << model->to_string() << " is added to the Pool" << std::endl;
}

template <>
void Manager::add(const ExecutableModel::Ptr& executable_model) {
    if (!tree_pool_.add<underlying_cast(TreePoolConfig::EXECUTABLE_MODEL_LEVEL)>(executable_model))
        throw std::runtime_error(
                "[Error] Fail to add a ExecutableModel(ID: 0x" + executable_model->get_id().to_string());
    ENN_DBG_COUT << executable_model->to_string() << " is added to the Pool" << std::endl;
}

template <>
void Manager::add(const ExecuteRequest::Ptr& execute_request) {
    if (!tree_pool_.add<underlying_cast(TreePoolConfig::EXECUTE_REQUEST_LEVEL)>(execute_request))
        throw std::runtime_error(
            "[Error] Fail to add a ExecuteRequest(ID: 0x" +
                execute_request->get_executable_model()->get_id().to_string());
    ENN_DBG_COUT << execute_request->to_string() << " is added to the Pool" << std::endl;
}

template <>
ClientProcess::Ptr Manager::get<ClientProcess>(FullIDType id) const {
    auto poolable = tree_pool_.find<underlying_cast(TreePoolConfig::CLIENT_PROCESS_LEVEL)>(ClientProcessID{id});
    if (!poolable)
        throw std::runtime_error(
                "[Error] Fail to get a ClientProcess(ID: 0x" + ClientProcessID{id}.to_string() + ")");
    return std::static_pointer_cast<ClientProcess>(poolable);
}

template <>
Model::Ptr Manager::get<Model>(FullIDType id) const {
    auto poolable = tree_pool_.find<underlying_cast(TreePoolConfig::MODEL_LEVEL)>(ModelID{id});
    if (!poolable)
        throw std::runtime_error(
                "[Error] Fail to get a Model(ID: 0x" + ModelID{id}.to_string() + ")");
    return std::static_pointer_cast<Model>(poolable);
}

template <>
ExecutableModel::Ptr Manager::get<ExecutableModel>(FullIDType id) const {
    auto poolable = tree_pool_.find<underlying_cast(TreePoolConfig::EXECUTABLE_MODEL_LEVEL)>(ExecutableModelID{id});
    if (!poolable)
        throw std::runtime_error(
                "[Error] Fail to get a ExecutableModel(ID: 0x" + ExecutableModelID{id}.to_string() + ")");
    return std::static_pointer_cast<ExecutableModel>(poolable);
}

template <>
ExecuteRequest::Ptr Manager::get<ExecuteRequest>(FullIDType id) const {
    auto poolable = tree_pool_.find<underlying_cast(TreePoolConfig::EXECUTE_REQUEST_LEVEL)>(ExecuteRequestID{id});
    if (!poolable)
        throw std::runtime_error(
                "[Error] Fail to get a ExecuteRequest(ID: 0x" + ExecuteRequestID{id}.to_string() + ")");
    return std::static_pointer_cast<ExecuteRequest>(poolable);
}

template <>
void Manager::release<ClientProcess>(FullIDType id) {
    ENN_DBG_COUT << "Try to release a ClientProcess(ID: 0x" << id <<
        ") from the Pool with Models, ExecutableModels and ExecuteRequests created from it..." << std::endl;
    if (!tree_pool_.remove<underlying_cast(TreePoolConfig::CLIENT_PROCESS_LEVEL)>(ClientProcessID{id}))
        throw std::runtime_error(
                "[Error] Fail to release a ClientProcess(ID: 0x" + ClientProcessID{id}.to_string() + ")");
    ENN_DBG_COUT << "Release done : "<< ClientProcessID{id}.to_string() << std::endl;
}

template <>
void Manager::release<Model>(FullIDType id) {
    ENN_DBG_COUT << "Try to release a Model(ID: 0x" << id <<
        ") from the Pool with ExecutableModels and ExecuteRequests created from it..." << std::endl;
    if (!tree_pool_.remove<underlying_cast(TreePoolConfig::MODEL_LEVEL)>(ModelID{id}))
        throw std::runtime_error(
                "[Error] Fail to release a Model(ID: 0x" + ModelID{id}.to_string() + ")");
    ENN_DBG_COUT << "Release done : "<< ModelID{id}.to_string() << std::endl;
}

template <>
void Manager::release<ExecutableModel>(FullIDType id) {
    ENN_DBG_COUT << "Try to release an ExecutableModel(ID: 0x" << id <<
        ") from the Pool..." << std::endl;
    if (!tree_pool_.remove<underlying_cast(TreePoolConfig::EXECUTABLE_MODEL_LEVEL)>(ExecutableModelID{id}))
        throw std::runtime_error(
                "[Error] Fail to release an ExecutableModel(ID: 0x" + ExecutableModelID{id}.to_string() + ")");
    ENN_DBG_COUT << "Release done : "<< ExecutableModelID{id}.to_string() << std::endl;
}

template <>
void Manager::release<ExecuteRequest>(FullIDType id) {
    ENN_DBG_COUT << "Try to release an ExecuteRequest from ExecutableModel(ID: 0x" << id <<
        ") from the Pool..." << std::endl;
    if (!tree_pool_.remove<underlying_cast(TreePoolConfig::EXECUTE_REQUEST_LEVEL)>(ExecuteRequestID{id}))
        throw std::runtime_error(
                "[Error] Fail to release an ExecuteRequest(ID: 0x" + ExecuteRequestID{id}.to_string() + ")");
    ENN_DBG_COUT << "Release done : "<< ExecuteRequestID{id}.to_string() << std::endl;
}

template <>
size_t Manager::count<ClientProcess>() const {
    return tree_pool_.width<underlying_cast(TreePoolConfig::CLIENT_PROCESS_LEVEL)>();
}

template <>
size_t Manager::count<Model>() const {
    return tree_pool_.width<underlying_cast(TreePoolConfig::MODEL_LEVEL)>();
}

template <>
size_t Manager::count<ExecutableModel>() const {
    return tree_pool_.width<underlying_cast(TreePoolConfig::EXECUTABLE_MODEL_LEVEL)>();
}

template <>
size_t Manager::count<ExecuteRequest>() const {
    return tree_pool_.width<underlying_cast(TreePoolConfig::EXECUTE_REQUEST_LEVEL)>();
}

};  // namespace pool
};  // namespace runtime
};  // namespace enn
