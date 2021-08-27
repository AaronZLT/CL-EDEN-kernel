#include "runtime/engine.hpp"
#include "runtime/userdriver_manager.hpp"
#include "model/model.hpp"
#include "model/parser/parser.hpp"
#include "model/raw/model.hpp"
#include "model/raw/data/operator.hpp"
#include "common/enn_memory_manager.h"
#include "model/generator/generator.hpp"
#include "runtime/pool/manager.hpp"
#include "runtime/scheduler/static_scheduler.hpp"
#include "tool/profiler/include/ExynosNnProfilerApi.h"
#include "runtime/execute_request/execute_request.hpp"
#include "runtime/client_process/client_process.hpp"
#include "common/enn_preference_generator.hpp"
#include "tool/dumper/frequency_dumper.hpp"
#include "tool/dumper/utilization_dumper.hpp"
#include "common/identifier_chopper.hpp"

#include <cinttypes>
#include <string>

namespace enn {
namespace runtime {

using namespace dispatch;

// Engine's implementation class
class Engine::EngineImpl {
 public:
    EngineImpl()
        : memory_manager_(std::make_unique<enn::EnnMemoryManager>()),
          userdriver_manager_(std::make_unique<UserdriverManager>()),
          model_pool_manager_(std::make_unique<pool::Manager>()) {
        memory_manager_->init();
    }
    ~EngineImpl() {
        memory_manager_->deinit();
    }
    EnnRet init();
    Engine::ModelID open_model(const LoadParameter& load_param, SessionBufInfo* session_info);
    Engine::ExecutableModelID commit_execution_data(const Engine::ModelID model_id, const InferenceData& exec_data);
    EnnRet execute_model(const std::vector<Engine::ExecutableModelID>& exec_id_list);
    EnnRet release_execution_data(Engine::ExecutableModelID exec_id);
    EnnRet close_model(Engine::ModelID model_id);
    EnnRet deinit();

    DeviceSessionID get_dsp_session_id(ModelID model_id);

    EnnRet shutdown_client_process(uint64_t client_pid);

private:
    inline model::ModelType read_model_type_from(const LoadParameter& load_param);
    std::vector<EnnBufferCore::Ptr> read_param_mem_infos_from(const std::vector<BufferCore> &params_in_model);
    void fill_session_info(SessionBufInfo* session_info, const model::Model::Ptr& enn_model);
    void create_execute_request();
    // Private members of EngineImpl have to provide thread safety since Engine is Singleton.
    // However, the following actors locally constructed and distructed to avoid data race condition.
    //  @ Actor classes locally created and depended by EngineImpl, which do not guarantee MT-safe.
    //    1. enn::model::Parser
    //    2. enn::model::Generator
    //    3. DispatcherImpl(OpenDispatcher, PrepareDispatcher, ExecuteDispatcher, CloseDispatcher)
    std::unique_ptr<enn::EnnMemoryManager> memory_manager_;
    UserdriverManager::UPtr userdriver_manager_;  // keeps userdriver instances
    pool::Manager::UPtr model_pool_manager_;
};

EnnRet Engine::EngineImpl::init() {
    ENN_INFO_PRINT("Init of engine is called from d %d\n", enn::util::get_caller_pid());
    ClientProcess::Ptr client_process = std::make_shared<ClientProcess>();
    try {
        model_pool_manager_->add(client_process);
    } catch(const std::exception& ex) {
        ENN_WARN_COUT << ex.what() << std::endl;
        ENN_WARN_COUT << "This process(ID: 0x" << client_process->get_id()
                     << ") is already initialized before." << std::endl;
    }

    return ENN_RET_SUCCESS;
}

EnnBufferCore::Ptr create_memory(std::unique_ptr<enn::EnnMemoryManager>& emm, const BufferCore& buffer) {
    enn::EnnMemoryManager::fd_type fd = (buffer.size) ? buffer.fd->data[0] : -1;
#ifdef ENN_MEDIUM_IF_HIDL
    if (fd < 0) {
        ENN_DBG_COUT << "Create dummy object" << std::endl;
        return emm->CreateMemoryObject(fd, 0, nullptr);
    }
    return emm->CreateMemoryFromFd(fd, buffer.size, buffer.fd.getNativeHandle());
#else
    // in case of non-hidl, {fd, va} can be directly used
    return emm->CreateMemoryObject(fd, buffer.size, reinterpret_cast<void*>(buffer.va));
#endif
}

Engine::ModelID Engine::EngineImpl::open_model(const LoadParameter& load_param, SessionBufInfo *session_info) {
    // 1. check if client process coming called init().
    ClientProcess::Ptr access_client = nullptr;
    try {
        access_client = model_pool_manager_->get<ClientProcess>(ClientProcess().get_id().get());
    } catch (const std::invalid_argument& ia) {
        ENN_ERR_COUT << ia.what() << std::endl;
        ENN_ERR_COUT << "open_model() is failed, because this client is not initialized." << std::endl;
        return 0;
    }

    // 2. Change Client's model memory to Runtime's model memory.
    //    Parser need 3 Parameters From Client's model memory.
    //    1) Model Type
    model::ModelType model_type = read_model_type_from(load_param);
    //    2) Model Memory
    auto memory_for_model = create_memory(memory_manager_, load_param.buf_load_model);
    std::shared_ptr<model::ModelMemInfo> model_mem_info = std::make_shared<model::ModelMemInfo>(memory_for_model->va,
                                                                                                memory_for_model->fd,
                                                                                                memory_for_model->size);
    //    3) Model's Param Memory (For CGO/CV model)
    auto param_mem_objs = read_param_mem_infos_from(load_param.buf_load_params);
    auto param_mem_infos = std::make_shared<std::vector<std::shared_ptr<enn::model::ModelMemInfo>>>();
    for (auto mem_obj : param_mem_objs) {
        param_mem_infos->push_back(
            std::make_shared<enn::model::ModelMemInfo>(mem_obj->va, mem_obj->fd, mem_obj->size, mem_obj->offset));
    }

    // 3. Parser to generate Raw Model From Memory.
    model::Parser parser;
    parser.Set(model_type, model_mem_info, param_mem_infos);
    auto raw_model = parser.Parse();

    // 4. Generator to generate Enn Model From Raw Model.
    model::Generator model_generator;
    enn::model::Model::Ptr enn_model;
    TRY {
        enn_model = model_generator.generate_model(raw_model, access_client);
    }
    CATCH(what) {
        ENN_ERR_COUT << "failed: " << what << std::endl;
        // Delete mem_loaded_model object created by memory_manager_->CreateMemory*(...)
        memory_manager_->DeleteMemory(memory_for_model);
        return 0;
    }
    END_TRY

    // Pass memory_object and memory_manager to release memory_object when model is released.
    for (auto &mem_obj : param_mem_objs)
        enn_model->add_memory_object(mem_obj);

    enn_model->add_memory_object(memory_for_model)
             ->set_memory_manager(memory_manager_.get());

    // 5. Static Schedule to Creat Op List for UD.
    enn::preference::EnnPreferenceGenerator pref_generator(load_param.preferences.u32_v);
    pref_generator.show();

    schedule::StaticScheduler static_scheduler;
    static_scheduler.set_model(enn_model)
                    .set_preset_id(pref_generator.get_preset_id())
                    .set_pref_mode(pref_generator.get_pref_mode())
                    .set_target_latency(pref_generator.get_target_latency())
                    .set_tile_num(pref_generator.get_tile_num())
                    .set_core_affinity(pref_generator.get_core_affinity())
                    .set_priority(pref_generator.get_priority())
                    .run();

    // 6. Dispatch Op List to UD.
    try {
        enn_model->set_open_dispatcher(userdriver_manager_->create_open_dispatcher())
                 ->set_close_dispatcher(userdriver_manager_->create_close_dispatcher())
                 ->load();
    } catch (const std::runtime_error& re) {
        ENN_ERR_COUT << re.what() << std::endl;
        ENN_ERR_COUT << "open_model() failed with dispatch to user driver" << std::endl;
        return 0;
    }

    // 7. Fill Session Info for Client.
    fill_session_info(session_info, enn_model);

    try {
        // 8. Add model to model pool via model pool manager.
        model_pool_manager_->add(enn_model);
    } catch (const std::exception& ex) {
        ENN_ERR_COUT << ex.what() << std::endl;
        ENN_ERR_COUT << "open_model() is failed" << std::endl;
        return 0;
    }

#ifdef UTILIZATION_DUMP
    // Dump only if the model is first opened.
    if (model_pool_manager_->count<model::Model>() == 1)
        dump::UtilizationDumper::get_instance()->start_dump();
#endif

#ifdef FREQUENCY_DUMP
    // Dump only if the model is first opened.
    if (model_pool_manager_->count<model::Model>() == 1)
        dump::FrequencyDumper::get_instance()->start_dump();
#endif

    // TODO(yc18.cho, TBD): Change the id to the ExecutableModelID after release_execution_data() is enabled.
    // Start to profile with model id
    START_PROFILER(enn_model->get_id().get());

    return enn_model->get_id();
}

Engine::ExecutableModelID Engine::EngineImpl::commit_execution_data(const Engine::ModelID model_id,
                                                                    const InferenceData& exec_data) {
    // NOTE(hoon98.choi): Please modify the below lines after implementation is done
    ENN_INFO_PRINT("called from pid %d\n", enn::util::get_caller_pid());
    ENN_INFO_PRINT("  - n_region: %d\n", exec_data.n_region);

    // ExecutableModel object to be created and returned.
    ExecutableModel::Ptr executable_model = nullptr;
    try {
        // Fetch model corresponding to mode id from pool
        auto prototype_model = model_pool_manager_->get<model::Model>(model_id);
        // create ExecutableModel
        executable_model = ExecutableModel::create(prototype_model)
                           ->set_memory_manager(memory_manager_.get());
        // TODO(daewhan.kim) : remove below after buffer meta data is redesigned.
        for (auto& data_ele : exec_data.inference_data) {
            ENN_INFO_PRINT("    - attr(%d), fd(%d), local_addr(0x%" PRIX64 "), size(%d), offset(%d)\n",
                            data_ele.exec_attr, data_ele.fd->data[0], data_ele.addr, data_ele.size, data_ele.offset);
#ifdef ENN_MEDIUM_IF_HIDL
            auto memory_for_executable_model = memory_manager_->CreateMemoryFromFd(
                data_ele.fd->data[0], data_ele.size, data_ele.fd.getNativeHandle());
#else
            // in case of non-hidl, {fd, va} can be directly used
            auto memory_for_executable_model =
                memory_manager_->CreateMemoryObject(
                    data_ele.fd->data[0], data_ele.size, reinterpret_cast<void *>(data_ele.addr));
#endif
            executable_model->add_memory_object(memory_for_executable_model);
        }
        // Build BufferTable and load it to Userdrivers by PrepareDispatcher
        executable_model->load(userdriver_manager_->create_prepare_dispatcher());
        // add ExecutableModel created to pool
        model_pool_manager_->add(executable_model);
    } catch (const std::exception& ex) {
        ENN_ERR_COUT << ex.what() << std::endl;
        ENN_ERR_COUT << "commit_execution_data() is failed" << std::endl;
        return 0;
    }

    try {
        // create ExecuteRequest by ExecutableModel and add it to pool.
        execute::ExecuteRequest::Ptr execute_request = execute::ExecuteRequest::create(executable_model);
        model_pool_manager_->add(std::move(execute_request));
    } catch (const std::exception& ex) {
        // remove ExecutableModel object from Pool to release to one created in this function.
        try {
            model_pool_manager_->release<ExecutableModel>(executable_model->get_id().get());
        } catch (const std::runtime_error& re) {
            ENN_ERR_COUT << re.what() << std::endl;
        }
        ENN_ERR_COUT << ex.what() << std::endl;
        ENN_ERR_COUT << "commit_execution_data() is failed" << std::endl;
        return 0;
    }
    // NOTE(hoon98.choi): zero means "Error". Please assign appropriate ID
    return executable_model->get_id();
}

EnnRet Engine::EngineImpl::execute_model(const std::vector<Engine::ExecutableModelID>& exec_id_list) {
    // NOTE(hoon98.choi): Currently only [0] is filled from API
    PROFILE_SCOPE("ExynosNN_Execution", util::chop_into_model_id(exec_id_list[0]));

    ENN_INFO_PRINT("Exec_id_list[0] = 0x%" PRIX64 "\n", exec_id_list[0]);
    try {
        model_pool_manager_->get<execute::ExecuteRequest>(exec_id_list[0])
                           ->execute(userdriver_manager_->create_execute_dispatcher());
    } catch (const std::exception& ex) {
        ENN_ERR_COUT << ex.what() << std::endl;
        ENN_ERR_COUT << "execute_model() is failed" << std::endl;
        return ENN_RET_FAILED;
    }
    return ENN_RET_SUCCESS;
}

EnnRet Engine::EngineImpl::release_execution_data(Engine::ExecutableModelID exec_id) {
    // TODO(hoon98.choi, TBD after exe_graph is done): implement: API -> mediuminterface -> IPC -> call this
    ENN_DBG_PRINT("Release Execution Data Start: ExecuteModelId[%ju]\n", exec_id);
    return ENN_RET_SUCCESS;
}

EnnRet Engine::EngineImpl::close_model(Engine::ModelID model_id) {
#ifdef UTILIZATION_DUMP
    // Dump only if the model is last closed.
    if (model_pool_manager_->count<model::Model>() == 1)
        dump::UtilizationDumper::get_instance()->finish_dump();
#endif

#ifdef FREQUENCY_DUMP
    // Dump only if the model is last closed.
    if (model_pool_manager_->count<model::Model>() == 1)
        dump::FrequencyDumper::get_instance()->finish_dump();
#endif
    // TODO(yc18.cho, TBD): Change the id to the ExecutableModelID after release_execution_data() is enabled.
    // Finish to profile with model id
    FINISH_PROFILER(model_id);

    ENN_INFO_PRINT(" received:  Model ID(0x%" PRIX64 ")\n", model_id);

    try {
        model_pool_manager_->release<model::Model>(model_id);
    } catch (const std::runtime_error& re) {
        ENN_ERR_COUT << re.what() << std::endl;
        ENN_ERR_COUT << "close_model() is failed, this model(ID: 0x"
                     << model_id << ") is not found in Pool." << std::endl;
        return ENN_RET_FAILED;
    }

    return ENN_RET_SUCCESS;
}

EnnRet Engine::EngineImpl::deinit() {
    ENN_INFO_PRINT("Deinit of engine is called from pid %d\n", enn::util::get_caller_pid());
    try {
        model_pool_manager_->release<ClientProcess>(ClientProcess().get_id().get());
    } catch(const std::runtime_error& re) {
        ENN_WARN_COUT << re.what() << std::endl;
        ENN_WARN_COUT << "This process(ID: 0x" << ClientProcess().get_id()
                     << ") is already deinitialized before." << std::endl;
    }

    return ENN_RET_SUCCESS;
}

EnnRet Engine::EngineImpl::shutdown_client_process(uint64_t client_pid) {
    ENN_INFO_PRINT(" received: Process ID(0x%" PRIX64 ")\n", client_pid);
    try {
        model_pool_manager_->release<ClientProcess>(client_pid << ClientProcess::UniqueID::Offset);
    } catch (const std::runtime_error& re) {
        ENN_WARN_COUT << re.what() << std::endl;
        ENN_WARN_COUT << "ClientProcess(0x" << std::hex << std::uppercase
                      << client_pid << ") is already released." << std::endl;
    }
    return ENN_RET_SUCCESS;
}

Buffer fill_user_buffer_info_by(const std::vector<model::metadata::BufferMetaData::Ptr>& meta_data_vector,
                                size_t meta_data_index, int32_t& prev_region_index) {
    Buffer user_buffer;
    auto& current_meta_data = meta_data_vector.at(meta_data_index);

    user_buffer.name = current_meta_data->get_name();
    user_buffer.size = current_meta_data->get_size();
    user_buffer.buf_index = current_meta_data->get_direction_idx();
    user_buffer.buffer_type = current_meta_data->get_data_type();
    user_buffer.region_idx = current_meta_data->get_region_index();
    user_buffer.shape.n = current_meta_data->get_shape()[N_NCHW];
    user_buffer.shape.c = current_meta_data->get_shape()[C_NCHW];
    user_buffer.shape.h = current_meta_data->get_shape()[H_NCHW];
    user_buffer.shape.w = current_meta_data->get_shape()[W_NCHW];

    if ((prev_region_index != current_meta_data->get_region_index()) || (meta_data_index == 0)) {
        user_buffer.offset = 0;
    } else {
        //  Same region index of Current & Prev data means Binding Buffer.
        size_t prev_user_buffer_size = meta_data_vector.at(meta_data_index - 1)->get_size();
        size_t prev_user_buffer_offset = meta_data_vector.at(meta_data_index - 1)->get_offset();
        current_meta_data->set_offset(prev_user_buffer_size + prev_user_buffer_offset);
        user_buffer.offset = current_meta_data->get_offset();
    }

    switch (current_meta_data->get_direction()) {
     case enn::model::Direction::Input :
        user_buffer.dir = DirType::ENN_BUF_DIR_IN;
        break;
     case enn::model::Direction::Output :
        user_buffer.dir = DirType::ENN_BUF_DIR_OUT;
        break;
     case enn::model::Direction::EXT :
        user_buffer.dir = DirType::ENN_BUF_DIR_EXT;
        break;
     case enn::model::Direction::None :
     case enn::model::Direction::SIZE :
     default :
        ENN_ERR_PRINT("Not Supported Buffer Direction : %d\n", (int)(current_meta_data->get_direction()));
        user_buffer.dir = DirType::ENN_BUF_DIR_NONE;
    }

    return user_buffer;
}

void fill_region_info_by(Buffer& current_user_buffer, SessionBufInfo* session_info,
                         size_t& region_count, int32_t& prev_region_index) {
    // Fill region when Current & Prev Region index is different.
    // If same, increase the size of Prev Region.
    if (current_user_buffer.region_idx != prev_region_index) {
        // TODO(yc18.cho&hoon98.choi, TBD): implement factory/setter functions for HIDL struct.
        Region region;
        region.req_size = current_user_buffer.size;
        region.attr = 0;  // initalized value

        session_info->regions[current_user_buffer.region_idx] = region;
        prev_region_index = current_user_buffer.region_idx;
        region_count++;
        ENN_DBG_COUT << "Region Created - id[" << current_user_buffer.region_idx << "] : size["
                        << region.req_size << "]" << std::endl;
    } else {
        session_info->regions[prev_region_index].req_size += current_user_buffer.size;
        ENN_DBG_COUT << "Region Request Size is increased - id[" << current_user_buffer.region_idx
                        << "] : size[" << session_info->regions[prev_region_index].req_size << "]" << std::endl;
    }
}

void Engine::EngineImpl::fill_session_info(SessionBufInfo* session_info, const model::Model::Ptr& enn_model) {
    ENN_DBG_COUT << "Start to fill Session Info of Model : " << enn_model->get_id() << std::endl;
    session_info->model_id = enn_model->get_id();

    size_t buffer_meta_data_num = enn_model->get_buffer_meta_data().size();
    session_info->buffers.resize(buffer_meta_data_num);
    session_info->regions.resize(buffer_meta_data_num);
    size_t region_count = 0;
    int32_t prev_region_idx = model::UNDEFINED;
    for (size_t i = 0; i < buffer_meta_data_num; i++) {

        // Fill Buffers
        Buffer user_buffer = fill_user_buffer_info_by(enn_model->get_buffer_meta_data(), i, prev_region_idx);
        session_info->buffers[i] = user_buffer;

        // Fill Regions
        fill_region_info_by(user_buffer, session_info, region_count, prev_region_idx);
    }

    if (buffer_meta_data_num != region_count) {
        session_info->regions.resize(region_count);
    }

    ENN_DBG_COUT << "Complete to fill Session Info."
                 << " Buffer Count : " << buffer_meta_data_num
                 << " Region Count : " << region_count
                 << std::endl;
}

Engine::DeviceSessionID Engine::EngineImpl::get_dsp_session_id(Engine::ModelID model_id) {
    ENN_DBG_COUT << "Model ID : " << model_id << std::endl;
    Engine::DeviceSessionID ret = 0;
    try {
        auto enn_model = model_pool_manager_->get<model::Model>(model_id);
        std::unique_ptr<SessionIdQueryDispatcher> session_id_query_dispatcher =
            userdriver_manager_->create_session_id_query_dispatcher();

        for (auto& opr_list : enn_model->get_scheduled_graph()->order<model::graph::iterator::BreadthFirstSearch>()) {
                auto op_list_session_info = std::make_unique<ExecutableOpListSessionInfo>(opr_list);
                session_id_query_dispatcher->dispatch(*op_list_session_info);
                ret = op_list_session_info->get_device_session_id();
                ENN_DBG_COUT << "Get Session ID From DSP User Driver : " << ret << std::endl;
        }
    } catch (const std::exception& e) {
        ENN_ERR_COUT << e.what() << std::endl;
        ENN_ERR_COUT << "get_dsp_session_id Failed" << std::endl;
        return 0;
    }
    return ret;
}

inline enn::model::ModelType Engine::EngineImpl::read_model_type_from(const LoadParameter& load_param) {
    return static_cast<enn::model::ModelType>(load_param.model_type);
}

std::vector<EnnBufferCore::Ptr> Engine::EngineImpl::read_param_mem_infos_from(
    const std::vector<BufferCore>& params_in_model) {
    ENN_DBG_COUT << "buf_load_params size : " << params_in_model.size() << std::endl;
    auto param_mem_objs = std::vector<EnnBufferCore::Ptr>();
    for (int p_idx = 0; p_idx < params_in_model.size(); ++p_idx) {
        auto&& param_ele = params_in_model[p_idx];
        if (param_ele.size != 0) {
            ENN_DBG_PRINT("FD: %d, va: %ju, size: %d\n", param_ele.fd->data[0], param_ele.va, param_ele.size);
        } else {
            ENN_DBG_PRINT("FD: N/A, va: %ju, size: %d\n", param_ele.va, param_ele.size);
        }
        auto mem = create_memory(memory_manager_, param_ele);
        param_mem_objs.push_back(mem);
    }
    ENN_DBG_COUT << "param_mem_objs size: " << param_mem_objs.size() << std::endl;
    return param_mem_objs;
}

std::mutex Engine::mutex_;
Engine* Engine::instance_ = nullptr;

Engine::Engine()
    : impl_(std::make_unique<EngineImpl>()) {}

// Destructor declaration is mendatory for compiler to know EngineImpl as complete type.
Engine::~Engine() = default;

Engine* Engine::get_instance() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (instance_ == nullptr) {
        instance_ = new Engine();
    }
    return instance_;
}

void Engine::destory_instance() {
    if (instance_ != nullptr) {
        delete instance_;
        instance_ = nullptr;
    }
}

EnnRet Engine::init() {
    return impl_->init();
}

EnnRet Engine::deinit() {
    return impl_->deinit();
}

Engine::ModelID Engine::open_model(const LoadParameter& load_param, SessionBufInfo *session_info) {
    return impl_->open_model(load_param, session_info);
}

Engine::ExecutableModelID Engine::commit_execution_data(const Engine::ModelID model_id,
                                                        const InferenceData& exec_data) {
    return impl_->commit_execution_data(model_id, exec_data);
}

Engine::EnnRet Engine::execute_model(const std::vector<Engine::ExecutableModelID>& exec_id_list) {
    return impl_->execute_model(exec_id_list);
}

Engine::EnnRet Engine::release_execution_data(Engine::ExecutableModelID exec_id) {
    return impl_->release_execution_data(exec_id);
}

Engine::EnnRet Engine::close_model(Engine::ModelID model_id) {
    return impl_->close_model(model_id);
}

Engine::EnnRet Engine::shutdown_client_process(uint64_t client_pid) {
    return impl_->shutdown_client_process(client_pid);
}

Engine::DeviceSessionID Engine::get_dsp_session_id(ModelID model_id) {
    return impl_->get_dsp_session_id(model_id);
}

};  // namespace runtime
};  // namespace enn
