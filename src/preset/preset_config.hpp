#ifndef SRC_PRESET_PRESET_CONFIG_HPP_
#define SRC_PRESET_PRESET_CONFIG_HPP_

#include <memory>
#include <mutex>
#include <vector>

#include "common/enn_common_type.h"
#include "common/enn_debug.h"
#include "common/enn_utils.h"

#include "jsoncpp/json/json.h"

//TODO(jinhyo01.kim) :: need to delete ( these const come from "drv_usr_if.h" in eden framework )
//#define NPU_S_PARAM_IS_PRESET 0x891000
//#define NPU_S_PARAM_QOS_RST (NPU_S_PARAM_IS_PRESET + 10)

//#define ENN_PRESET_VAR_MIN (NPU_S_PARAM_IS_PRESET - NPU_S_PARAM_IS_PRESET)
//#define ENN_PRESET_VAR_MAX (NPU_S_PARAM_QOS_RST - NPU_S_PARAM_IS_PRESET)
//#define PRESET_MAX_PRIORITY	100
//#define PRESET_MIN_PRIORITY	1
//#define PRESET_LOCAL_ID_MIN 10000

namespace enn {
namespace preset {

/*
enum PresetType_e {
ENN_PRESET_TYPE_OPEN,
ENN_PRESET_TYPE_EXEC,
ENN_PRESET_TYPE_MAX
};
*/
static constexpr uint32_t preset_max_priority = 100;
static constexpr uint32_t preset_min_priority = 1;
// user can set preset_id from 0x0001 to 0xFFFF
static constexpr uint32_t preset_user_id_min = 0x00000001;
static constexpr uint32_t preset_user_id_max = 0x0000FFFF;
// PresetConfigManager can set preset_id automatically from 0x00010000 to 0xFFFF0000
// And PresetConfigManager increase preset_id by 0x00010000
static constexpr uint32_t preset_local_id_min = 0x00010000;
static constexpr uint32_t preset_local_id_max = 0xFFFF0000;
static constexpr uint32_t preset_local_id_increment = 0x00010000;

static constexpr char* preset_name_prefix = const_cast<char *>("PRESET_");

static constexpr int32_t preset_var_not_used = -1;

enum class PresetType_e {
    // currently supporting below 2 types
    ENN_PRESET_TYPE_NONE, // not used
    ENN_PRESET_TYPE_OPEN_CLOSE,
    ENN_PRESET_TYPE_EXEC_ONLY,
    ENN_PRESET_TYPE_INIT_DEINIT,
    //TODO (jinhyo01.kim) : preset policies need to be implemented for below enums
    ENN_PRESET_TYPE_INIT_ONLY,
    ENN_PRESET_TYPE_OPEN_ONLY,
    ENN_PRESET_TYPE_CLOSE_ONLY,
    ENN_PRESET_TYPE_DEINIT_ONLY,
    ENN_PRESET_TYPE_MAX=ENN_PRESET_TYPE_INIT_DEINIT,
    ENN_PRESET_TYPE_MIN=ENN_PRESET_TYPE_NONE,
};

//TODO (jinhyo01.kim) :: need to use enum prefer_target in userdriver/npu/drv_usr_if.h
enum class PresetVars_e {
    /* legacy target */
    ENN_PRESET_VAR_NPU_FREQ = 0,
    ENN_PRESET_VAR_DNC_FREQ,
    ENN_PRESET_VAR_DSP_FREQ,
    ENN_PRESET_VAR_MIF_FREQ,
    ENN_PRESET_VAR_INT_FREQ,
    ENN_PRESET_VAR_CPU_LIT_FREQ,
    ENN_PRESET_VAR_CPU_MID_FREQ,
    ENN_PRESET_VAR_CPU_BIG_FREQ,
    ENN_PRESET_VAR_MO_SCEN,
    /* added in exynos9925 */
    ENN_PRESET_VAR_CPU_AFF,
    ENN_PRESET_VAR_DD_KPI_MODE,
    ENN_PRESET_VAR_APP_ID,
    ENN_PRESET_VAR_MODEL_ID,
    ENN_PRESET_VAR_SUBGRAPH_ID,
    ENN_PRESET_VAR_MODEL_NAME,
    ENN_PRESET_VAR_GPU_FREQ,
    ENN_PRESET_VAR_CPU_LIT_IDLE,
    ENN_PRESET_VAR_CPU_MID_IDLE,
    ENN_PRESET_VAR_CPU_BIG_IDLE,
    ENN_PRESET_VAR_FW_HINT,
    ENN_PRESET_VAR_LLC,
    ENN_PRESET_VAR_LLC_SCEN,
    ENN_PRESET_VAR_MAX = ENN_PRESET_VAR_LLC_SCEN,
    ENN_PRESET_VAR_MIN = ENN_PRESET_VAR_NPU_FREQ,
};

static std::map<std::string, PresetVars_e> preset_type_name_map_ = {
    {"NPU_FREQ", PresetVars_e::ENN_PRESET_VAR_NPU_FREQ},
    {"DNC_FREQ", PresetVars_e::ENN_PRESET_VAR_DNC_FREQ},
    {"DSP_FREQ", PresetVars_e::ENN_PRESET_VAR_DSP_FREQ},
    {"MIF_FREQ", PresetVars_e::ENN_PRESET_VAR_MIF_FREQ},
    {"INT_FREQ", PresetVars_e::ENN_PRESET_VAR_INT_FREQ},
    {"CPU_LIT_FREQ", PresetVars_e::ENN_PRESET_VAR_CPU_LIT_FREQ},
    {"CPU_MID_FREQ", PresetVars_e::ENN_PRESET_VAR_CPU_MID_FREQ},
    {"CPU_BIG_FREQ", PresetVars_e::ENN_PRESET_VAR_CPU_BIG_FREQ},
    {"MO_SCENARIO", PresetVars_e::ENN_PRESET_VAR_MO_SCEN},
    {"CPU_AFFINITY", PresetVars_e::ENN_PRESET_VAR_CPU_AFF},
    {"DD_KPI_MODE", PresetVars_e::ENN_PRESET_VAR_DD_KPI_MODE},

    //TODO (jinhyo01.kim) : delete APP_ID ~ SUBGRAPH_ID ? for debugging
    {"APP_ID", PresetVars_e::ENN_PRESET_VAR_APP_ID},
    {"MODEL_ID", PresetVars_e::ENN_PRESET_VAR_MODEL_ID},
    {"SUBGRAPH_ID", PresetVars_e::ENN_PRESET_VAR_SUBGRAPH_ID},

    {"GPU_FREQ", PresetVars_e::ENN_PRESET_VAR_GPU_FREQ},
    {"CPU_LIT_IDLE", PresetVars_e::ENN_PRESET_VAR_CPU_LIT_IDLE},
    {"CPU_MID_IDLE", PresetVars_e::ENN_PRESET_VAR_CPU_MID_IDLE},
    {"CPU_BIG_IDLE", PresetVars_e::ENN_PRESET_VAR_CPU_BIG_IDLE},
    {"FW_HINT", PresetVars_e::ENN_PRESET_VAR_FW_HINT},
    {"LLC", PresetVars_e::ENN_PRESET_VAR_LLC},
    {"LLC_SCEN", PresetVars_e::ENN_PRESET_VAR_LLC_SCEN},
};

//TODO (jinhyo01.kim) : select one vector of below two vectors
static std::vector<std::string> preset_type_name_vector_ = {
    "NPU_FREQ",
    "DNC_FREQ",
    "DSP_FREQ",
    "MIF_FREQ",
    "INT_FREQ",
    "CPU_LIT_FREQ",
    "CPU_MID_FREQ",
    "CPU_BIG_FREQ",
    "MO_SCENARIO",
    "CPU_AFFINITY"
};

static std::vector<std::string> preset_type_name_vector_v2_ = {
    "NPU_FREQ",
    "DNC_FREQ",
    "DSP_FREQ",
    "MIF_FREQ",
    "INT_FREQ",
    "CPU_LIT_FREQ",
    "CPU_MID_FREQ",
    "CPU_BIG_FREQ",
    "MO_SCENARIO",
    "CPU_AFFINITY",
    "DD_KPI_MODE",
    "APP_ID",
    "MODEL_ID",
    "SUBGRAPH_ID",
    "GPU_FREQ",
    "CPU_LIT_IDLE",
    "CPU_MID_IDLE",
    "CPU_BIG_IDLE",
    "FW_HINT",
    "LLC",
    "LLC_SCEN"
};

class PresetConfig {
    using ModelId = EnnModelId;
    using EnnExecuteModelId = EnnModelId;
    using EnnRet = EnnReturn;

public:
    PresetConfig() = default;
    PresetConfig(uint32_t preset_id) : PresetId_(preset_id){};
    PresetConfig(uint32_t preset_id, std::string preset_name) : PresetId_(preset_id), PresetName_(preset_name){};
    ~PresetConfig() = default;

    void show();
    void show_v2();
    void dump(void);

    uint32_t get_preset_id(){
        return PresetId_;
    }
    std::string get_preset_name(){
        return PresetName_;
    }
    int32_t get_target_cpu_lit_freq(void) { return PresetVars_[static_cast<int>(PresetVars_e::ENN_PRESET_VAR_CPU_LIT_FREQ)]; }
    int32_t get_target_cpu_mid_freq(void) { return PresetVars_[static_cast<int>(PresetVars_e::ENN_PRESET_VAR_CPU_MID_FREQ)]; }
    int32_t get_target_cpu_big_freq(void) { return PresetVars_[static_cast<int>(PresetVars_e::ENN_PRESET_VAR_CPU_BIG_FREQ)]; }
    int32_t get_target_npu_freq(void) { return PresetVars_[static_cast<int>(PresetVars_e::ENN_PRESET_VAR_NPU_FREQ)]; }
    int32_t get_target_dnc_freq(void) { return PresetVars_[static_cast<int>(PresetVars_e::ENN_PRESET_VAR_DNC_FREQ)]; }
    int32_t get_target_dsp_freq(void) { return PresetVars_[static_cast<int>(PresetVars_e::ENN_PRESET_VAR_DSP_FREQ)]; }
    int32_t get_target_mif_freq(void) { return PresetVars_[static_cast<int>(PresetVars_e::ENN_PRESET_VAR_MIF_FREQ)]; }
    int32_t get_target_int_freq(void) { return PresetVars_[static_cast<int>(PresetVars_e::ENN_PRESET_VAR_INT_FREQ)]; }
    int32_t get_target_dd_kpi_mode(void) { return PresetVars_[static_cast<int>(PresetVars_e::ENN_PRESET_VAR_DD_KPI_MODE)]; }

    EnnReturn set_preset_id(uint32_t id){
        if( id > 0){
            PresetId_ = id;
            return ENN_RET_SUCCESS;
        }else{
            return ENN_RET_FAILED;
        }
    }

    EnnReturn set_preset_name(std::string name){
        if( name.size() > 0){
            PresetName_ = name;
            return ENN_RET_SUCCESS;
        }else{
            return ENN_RET_FAILED;
        }
    }

    friend class PresetConfigManager;

private:
    /* old variables */
    uint32_t PresetId_;
    int32_t PresetVars_[static_cast<int>(PresetVars_e::ENN_PRESET_VAR_MAX)];
    uint32_t CpuAffinityMask_;
    uint32_t PresetTypeScenario_[static_cast<int>(PresetType_e::ENN_PRESET_TYPE_MAX)];

    // New ENN feature
    uint32_t PresetPriority_[static_cast<int>(PresetVars_e::ENN_PRESET_VAR_MAX)]; // default value is MIN_PRESET_PRIORITY(=1)
    std::string PresetName_;
    std::map<std::string, int32_t> PresetVars_map_;

}; // end of PresetCofig

class PresetConfigManager {
    using ModelId = EnnModelId;
    using EnnExecuteModelId = EnnModelId;
    using EnnRet = EnnReturn;

public:
    PresetConfigManager();
    ~PresetConfigManager() = default;

    static PresetConfigManager *get_instance();
    static void destory_instance();

    void show();

    //TODO (jinhyo01.kim) : select one function of below three functions
    EnnReturn parse_scenario_from_ext_json_old(const std::string preset_json_file);
    EnnReturn parse_scenario_from_ext_json(const std::string preset_json_file);
    EnnReturn parse_scenario_from_ext_json_v2(const std::string preset_json_file);

    //TODO (jinhyo01.kim) : delete below functions if not needed
    EnnReturn get_id_from_ext_json(std::string ncpName, std::shared_ptr<uint32_t> id);
    EnnReturn get_enn_preset_external_scenario(const uint32_t key, std::shared_ptr<PresetConfig *> preset_config);
    EnnReturn get_enn_preset_scenario(uint32_t presetId, std::shared_ptr<PresetConfig *> preset_config);
    std::shared_ptr<PresetConfig> get_preset_config_by_id(uint32_t presetId);
    EnnReturn delete_enn_preset_scenario();
    EnnReturn get_preset_id(std::string name, std::shared_ptr<uint32_t> presetId);
    EnnReturn reset_local_id();

    uint32_t get_local_id();

    std::string get_local_name(uint32_t id);

    uint32_t get_preset_id_by_name(std::string name);

    EnnReturn parse_preset_id(const enn::preset::json::Value &element, std::shared_ptr<PresetConfig> preset_config);
    EnnReturn parse_preset_name(const enn::preset::json::Value &element, std::shared_ptr<PresetConfig> preset_config);
    EnnReturn parse_preset_vars(const enn::preset::json::Value &element, std::shared_ptr<PresetConfig> preset_config);
    EnnReturn parse_preset_type(const enn::preset::json::Value &element, std::shared_ptr<PresetConfig> preset_config);
    EnnReturn parse_preset_priority(const enn::preset::json::Value &element, std::shared_ptr<PresetConfig> preset_config);

private:
    uint32_t preset_local_id_counter_;
    std::map<uint32_t, std::shared_ptr<PresetConfig>> preset_config_map_by_id_;
    std::map<std::string, std::shared_ptr<PresetConfig>> preset_config_map_by_name_;

    static std::mutex mutex_;
    static PresetConfigManager *instance_;

}; // end of PresetConfigManager
}; // namespace preset
}; // namespace enn

#endif // SRC_PRESET_PRESET_CONFIG_HPP_
