#ifndef SRC_RUNTIME_ENGINE_HPP_
#define SRC_RUNTIME_ENGINE_HPP_

#include <mutex>
#include <vector>

#include "common/enn_common_type.h"
#include "common/enn_debug.h"

#ifdef ENN_MEDIUM_IF_HIDL
#include "medium/types.hal.hidl.h"
#else
#include "medium/types.hal.h"
using namespace enn::hal;
#endif

namespace enn {
namespace runtime {

class Engine {
    using ModelID = uint64_t;
    using ExecutableModelID = uint64_t;
    using EnnRet = EnnReturn;
    using DeviceSessionID = int32_t;  // For DSP only.

public:
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    // It returns Singleton which is initialized once (MT-safe).
    // It will be called when end-user calls EnnInitialize API.
    static Engine* get_instance();

    // It should be called when end-user calls EnnShutdown API.
    static void destory_instance();

    // It initializes by the process id of the client's side.
    EnnRet init();
    // It opens model taking memory address in which the model is loaded.
    // It will return model id that is unique to identify open models.
    // Second parameter is container that has end in/output buffer's metadata
    //  such as the number of buffers and each size.
    //  Client would give this pointer as empty, and then Engine analyzes the model and fills in
    //  the necessary buffer's metadata.
    ModelID open_model(const LoadParameter& load_param, SessionBufInfo *session_info);

    // It returns unique and identifiable id of a executable model.
    // The executable model is object injected with buffers of feature maps as well as
    //  static information of model opened.
    //  To explain in detail, the static data of the model that does not change after being opened like the operator
    //  and the memory buffer information of feature-maps that can be dynamically injected at each execution
    //  are all collected into one set and frozen.
    //  In other words, the executable model is a set with all the information needed to execute the model.
    // Second parameter is buffer table that contains information of end in/output buffers client allocated.
    //  Client should allocate these in/out buffers according to container of buffer's metadata
    //  which open_model API informed before.
    ExecutableModelID commit_execution_data(const EnnModelId model_id, const InferenceData& exec_data);

    // It executes model as taking an id of executable model.
    //  Second parameter should be end output buffer's array so that Engine can set the result of execution.
    EnnRet execute_model(const std::vector<ExecutableModelID>& exec_id_list);

    // It release executable model loaded by load_executable_model API function.
    //  Except for the static data of the model, all dynamically changing objects such as memory buffers are released.
    EnnRet release_execution_data(ExecutableModelID exec_id);

    // It close model corresponding to model id given by end-user.
    //  It releases and destroys data and objects related to given model including the static data of the model.
    EnnRet close_model(ModelID model_id);

    // It deinitializes by the process id of the client's side.
    EnnRet deinit();

    // It query DSP device session id for opened model via model id.
    DeviceSessionID get_dsp_session_id(ModelID model_id);

    EnnRet shutdown_client_process(uint64_t client_pid);

private:
    explicit Engine();
    ~Engine();

    class EngineImpl;
    std::unique_ptr<EngineImpl> impl_;

    static std::mutex mutex_;
    static Engine* instance_;
};

};  // namespace runtime
};  // namespace enn

#endif  // SRC_RUNTIME_ENGINE_HPP_
