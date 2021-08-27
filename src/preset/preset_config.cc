#include <cinttypes>
#include <string>
#include "preset_config.hpp"
#include "common/enn_debug.h"

#include <fstream>
#include <stdlib.h>

namespace enn {
namespace preset {

/*
 * PresetConfig Class Functions
 */

void PresetConfig::show(){
    std::cout << "Preset id : " << std::dec << PresetId_ << " (0x" << std::hex << PresetId_ << std::dec << ")" << std::endl;
    std::cout << "Preset name : " << PresetName_ << std::endl;
    std::cout << "Preset Vars = { " << std::endl;

    //PresetVar_[]
    int i = 0;
    for( auto iter = preset_type_name_vector_.begin(); iter != preset_type_name_vector_.end(); iter++){
        std::cout << "{ " << *iter << " : " << PresetVars_[i++] << " }" << std::endl;
    }
    std::cout << "}" << std::endl;

    //PresetTypeScenario_[]
    for( int j = 0 ; j < static_cast<int>(PresetType_e::ENN_PRESET_TYPE_MAX) ; j++){
        std::cout << "[ ";
        i = 0;
        for( auto iter = preset_type_name_vector_.begin(); iter != preset_type_name_vector_.end(); iter++){
            int k = (PresetTypeScenario_[j] >> (i++)) & 1;
            if( iter != preset_type_name_vector_.begin())
                std::cout << " , " ;
            std::cout << k;
        }
        std::cout << " ]" << std::endl;
    }
}

void PresetConfig::show_v2(){
    std::cout << "Preset id : "  << std::dec << PresetId_ << " (0x" << std::hex << PresetId_ << ")"  << std::dec << std::endl;
    std::cout << "Preset name : " << PresetName_ << std::endl;

    std::cout << "Preset Vars = { " << std::endl;

    //PresetVar_[]
    int i = 0;
    for( auto iter = preset_type_name_vector_v2_.begin(); iter != preset_type_name_vector_v2_.end(); iter++){
        std::cout << "{ " << *iter << " : " << PresetVars_[i++] << " }" << std::endl;
    }
    std::cout << "}" << std::endl;

    //PresetTypeScenario_[]
    for( int j = 0 ; j < static_cast<int>(PresetType_e::ENN_PRESET_TYPE_MAX) ; j++){
        std::cout << "[ ";
        i = 0;
        for( auto iter = preset_type_name_vector_v2_.begin(); iter != preset_type_name_vector_v2_.end(); iter++){
            int k = (PresetTypeScenario_[j] >> (i++)) & 1;
            if( iter != preset_type_name_vector_v2_.begin())
                std::cout << " , " ;
            std::cout << k;
        }
        std::cout << " ]" << std::endl;
    }
}

void PresetConfig::dump() {
    ENN_DBG_PRINT("Preset id:%u(%x)\n", PresetId_, PresetId_);
    ENN_DBG_PRINT("Preset name: %s\n", PresetName_.c_str());

    int i = 0;
    for (auto iter = preset_type_name_vector_v2_.begin(); iter != preset_type_name_vector_v2_.end(); iter++, i++)
        ENN_DBG_PRINT("PresetVars %s:\t%u\n", iter->c_str(), PresetVars_[i]);
}

/*
 * PresetConfigManager Class VariablesFunctions
 */

std::mutex PresetConfigManager::mutex_;
PresetConfigManager* PresetConfigManager::instance_ = nullptr;

PresetConfigManager* PresetConfigManager::get_instance() {
    std::lock_guard<std::mutex> guard(mutex_);
    if(instance_ == nullptr){
        instance_ = new PresetConfigManager();
    }
    return instance_;
}

void PresetConfigManager::destory_instance() {
    if (instance_ != nullptr) {
        delete instance_;
        instance_ = nullptr;
    }
}

PresetConfigManager::PresetConfigManager(){
    preset_local_id_counter_ = enn::preset::preset_local_id_min;
}

uint32_t PresetConfigManager::get_local_id(){
    std::lock_guard<std::mutex> guard(mutex_);
    preset_local_id_counter_ += enn::preset::preset_local_id_increment;
    if(preset_local_id_counter_ > enn::preset::preset_local_id_max)
        preset_local_id_counter_ = enn::preset::preset_local_id_min;

    return preset_local_id_counter_;
}

std::string PresetConfigManager::get_local_name(uint32_t id){
    return enn::preset::preset_name_prefix + std::to_string(id);
}

EnnReturn PresetConfigManager::parse_scenario_from_ext_json_old(const std::string preset_json_file) {

    enn::preset::json::Value root;

    std::ifstream ifs(preset_json_file);

    if (ifs.is_open()) {
        ifs >> root;
    } else {
        ENN_ERR_PRINT("File doesn't exist: %s\n", preset_json_file.c_str());
        return ENN_RET_FAILED;
    }

    /* parsing from json file */
    for (const enn::preset::json::Value &element : root["external_presets"]) {
        uint32_t key_id = element["PresetId"].asInt();
        ENN_DBG_PRINT(" # PresetId : %d\n", key_id);

        //std::shared_ptr<PresetConfig> tmp_preset = std::make_shared<PresetConfig(key_id)>;
        std::shared_ptr<PresetConfig> tmp_preset(new PresetConfig(key_id));
        auto &ele_PresetVars = element["PresetVars"];
        for (int i = 0; i < ele_PresetVars.size(); i++)
            tmp_preset->PresetVars_[i] = ele_PresetVars[i].asInt();
        auto &ele_PresetTypeScenario = element["PresetTypeScenario"];

        /* bit oring */
        uint32_t TypeScenario[static_cast<int>(PresetType_e::ENN_PRESET_TYPE_MAX)] = {};
        for (int i = 0; i < ele_PresetTypeScenario.size(); i++) {
            for (int j = 0; j < ele_PresetTypeScenario[i].size(); j++) {
                if (ele_PresetTypeScenario[i][j].asInt()) {
                    /* Range nested is overwritten by range containing it */
                    bool overlap = false;
                    for (int k = 0; k < i; k++)
                        if (tmp_preset->PresetTypeScenario_[k] & (1 << j)) {
                            overlap = true;
                            break;
                        }
                    if (!overlap) {
                        TypeScenario[i] |= (1 << j);
                    }
                }
            }
            tmp_preset->PresetTypeScenario_[i] = TypeScenario[i];
        }
        tmp_preset->show();

        preset_config_map_by_id_[key_id] = tmp_preset;
    }
    ifs.close();

    return ENN_RET_SUCCESS;
}

//TODO (jinhyo01.kim) : add a handling code for parsing error
EnnReturn PresetConfigManager::parse_scenario_from_ext_json(const std::string preset_json_file) {
    enn::preset::json::Value root;

    std::ifstream ifs(preset_json_file);

    if (ifs.is_open()) {
        ifs >> root;
    } else {
        ENN_ERR_PRINT("File doesn't exist: %s\n", preset_json_file.c_str());
        return ENN_RET_FAILED;
    }

    /* parsing from json file */
    for (const enn::preset::json::Value &element : root["external_presets"]) {
        uint32_t key_id = element["PresetId"].asInt();
        std::string preset_name = element["PresetName"].asString();

        if( (key_id < preset_user_id_min) || (key_id > preset_user_id_max)){
            ENN_WARN_PRINT("user preset id (%d) is not available!", key_id);
            key_id = get_local_id();
            ENN_WARN_PRINT("new preset id = %d\n", key_id);
        }
        ENN_DBG_PRINT(" # PresetId : %d\n", key_id);
        ENN_DBG_PRINT(" # PresetName : [%s]\n", preset_name.c_str());

        if(preset_name.empty()){
            ENN_DBG_PRINT(" # preset_name is empty \n");
            preset_name = element["name"].asString();
            ENN_DBG_PRINT(" # name : [%s]\n", preset_name.c_str());
            if(preset_name.empty()){
                ENN_DBG_PRINT(" # preset_name is empty \n");
                //TODO (jinhyo01.kim) : add a seperate function such as get_preset_name()
                //preset_name = enn::preset::preset_name_prefix + std::to_string(key_id);
                preset_name = get_local_name(key_id);
            }
        }
        //TODO (jinhyo01.kim) : apply a policy when there is a conflict in preset_id
        if( preset_config_map_by_id_.count(key_id) > 0){
            ENN_WARN_PRINT("preset id (%d) is already in preset json file\n", key_id);
        }
        //TODO (jinhyo01.kim) : apply a policy when there is a conflict in preset_name
        if( preset_config_map_by_name_.count(preset_name) > 0){
            ENN_WARN_PRINT("preset name (%s) is already in preset json file\n", preset_name.c_str());
        }

        std::shared_ptr<PresetConfig> tmp_preset(new PresetConfig(key_id, preset_name));
        auto &ele_PresetVars = element["PresetVars"];
        for (int i = 0; i < ele_PresetVars.size(); i++)
            tmp_preset->PresetVars_[i] = ele_PresetVars[i].asInt();

        auto &ele_PresetTypeScenario = element["PresetTypeScenario"];
        /* bit oring */
        //TODO (jinhyo01.kim) : add a seperate function for bit oring
        uint32_t TypeScenario[static_cast<int>(PresetType_e::ENN_PRESET_TYPE_MAX)] = {};
        for (int i = 0; i < ele_PresetTypeScenario.size(); i++) {
            for (int j = 0; j < ele_PresetTypeScenario[i].size(); j++) {
                if (ele_PresetTypeScenario[i][j].asInt()) {
                    /* Range nested is overwritten by range containing it */
                    bool overlap = false;
                    for (int k = 0; k < i; k++)
                        if (tmp_preset->PresetTypeScenario_[k] & (1 << j)) {
                            overlap = true;
                            break;
                        }
                    if (!overlap) {
                        TypeScenario[i] |= (1 << j);
                    }
                }
            }
            tmp_preset->PresetTypeScenario_[i] = TypeScenario[i];
        }
        tmp_preset->show();

        preset_config_map_by_id_.insert(make_pair(key_id, tmp_preset));
        preset_config_map_by_name_.insert(make_pair(preset_name, tmp_preset));

    }
    ifs.close();

    return ENN_RET_SUCCESS;
}

EnnReturn PresetConfigManager::parse_preset_id(const enn::preset::json::Value &element, std::shared_ptr<PresetConfig> preset_config){
    uint32_t preset_id = element["PresetId"].asInt();
    if( (preset_id < preset_user_id_min) || (preset_id > preset_user_id_max)){
        ENN_WARN_PRINT("user preset id (%d) is not available!", preset_id);
        preset_id = get_local_id();
        ENN_WARN_PRINT("new preset id = %d\n", preset_id);
    }
    //TODO (jinhyo01.kim) : apply a policy when there is a conflict in preset_id
    if( preset_config_map_by_id_.count(preset_id) > 0){
        ENN_WARN_PRINT("preset id (%d) is already in preset json file\n", preset_id);
    }
    return preset_config->set_preset_id(preset_id);
}

EnnReturn PresetConfigManager::parse_preset_name(const enn::preset::json::Value &element, std::shared_ptr<PresetConfig> preset_config){
    std::string preset_name = element["PresetName"].asString();
    if(preset_name.empty()){
        ENN_DBG_PRINT(" # preset_name is empty \n");
        preset_name = element["name"].asString();
        ENN_DBG_PRINT(" # name : [%s]\n", preset_name.c_str());
        if(preset_name.empty()){
            ENN_DBG_PRINT(" # preset_name is empty \n");
            preset_name = get_local_name(preset_config->get_preset_id());
        }
    }
    //TODO (jinhyo01.kim) : apply a policy when there is a conflict in preset_name
    if( preset_config_map_by_name_.count(preset_name) > 0){
        ENN_WARN_PRINT("preset name (%s) is already in preset json file\n", preset_name.c_str());
    }

    return preset_config->set_preset_name(preset_name);
}

EnnReturn PresetConfigManager::parse_preset_vars(const enn::preset::json::Value &element, std::shared_ptr<PresetConfig> preset_config){
    const enn::preset::json::Value &ele_PresetVars = element["PresetVars"];
    for (auto iter = preset_type_name_map_.begin() ; iter != preset_type_name_map_.end(); iter++){
        ENN_DBG_PRINT("PresetVars::Name = %s }\n", iter->first.c_str());
        int32_t tmp_preset_var = ele_PresetVars[iter->first].asInt();
        ENN_DBG_PRINT("Value = %d }\n", tmp_preset_var);

        if(tmp_preset_var != 0 )
            preset_config->PresetVars_[static_cast<int>(iter->second)] = tmp_preset_var;
        else
            preset_config->PresetVars_[static_cast<int>(iter->second)] = enn::preset::preset_var_not_used;
        ENN_DBG_PRINT("{ %s : %d }\n", iter->first.c_str(), tmp_preset_var);
    }
    return ENN_RET_SUCCESS;
}

EnnReturn PresetConfigManager::parse_preset_type(const enn::preset::json::Value &element, std::shared_ptr<PresetConfig> preset_config){
    uint32_t tmp_preset_type = element["PresetType"].asInt();
    ENN_DBG_PRINT("PresetType = %d }\n", tmp_preset_type);
    //PresetType : 1 - all preset scenario is ENN_PRESET_TYPE_OPEN_CLOSE
    //PresetType : 2 - all preset scenario is ENN_PRESET_TYPE_EXEC_ONLY
    if( tmp_preset_type > 0){
        if((tmp_preset_type < static_cast<int>(PresetType_e::ENN_PRESET_TYPE_MIN))
        || (tmp_preset_type > static_cast<int>(PresetType_e::ENN_PRESET_TYPE_MAX)))
            ENN_WARN_PRINT("preset_type (%d) is not available!\n", tmp_preset_type);
        else
            preset_config->PresetTypeScenario_[tmp_preset_type] = 0xFFFFFFFF;
    }else {
        auto &ele_PresetTypeScenario = element["PresetTypeScenario"];
        // bit oring
        uint32_t TypeScenario[static_cast<int>(PresetType_e::ENN_PRESET_TYPE_MAX)-static_cast<int>(PresetType_e::ENN_PRESET_TYPE_MIN)] = {};
        for (int i = 0; i < ele_PresetTypeScenario.size(); i++) {
            for (int j = 0; j < ele_PresetTypeScenario[i].size(); j++) {
                if (ele_PresetTypeScenario[i][j].asInt()) {
                    // Range nested is overwritten by range containing it
                    bool overlap = false;
                    for (int k = 0; k < i; k++)
                        if (preset_config->PresetTypeScenario_[k] & (1 << j)) {
                            overlap = true;
                            break;
                        }
                    if (!overlap) {
                        TypeScenario[i] |= (1 << j);
                    }
                }
            }
            preset_config->PresetTypeScenario_[i] = TypeScenario[i];
        }
    }
    return ENN_RET_SUCCESS;
}

EnnReturn PresetConfigManager::parse_preset_priority(const enn::preset::json::Value &element, std::shared_ptr<PresetConfig> preset_config){
    uint32_t tmp_preset_priority = element["PresetPriority"].asInt();
    if( (tmp_preset_priority < enn::preset::preset_min_priority) || (tmp_preset_priority > enn::preset::preset_max_priority)){
        ENN_WARN_PRINT("preset priority (%d) is not available!\n", tmp_preset_priority);
        return ENN_RET_FAILED;
    }

    for (auto iter = preset_type_name_map_.begin() ; iter != preset_type_name_map_.end(); iter++){
        preset_config->PresetPriority_[static_cast<int>(iter->second)] = tmp_preset_priority;
    }
    return ENN_RET_SUCCESS;
}

//TODO (jinhyo01.kim) : add a handling code for parsing error
EnnReturn PresetConfigManager::parse_scenario_from_ext_json_v2(const std::string preset_json_file) {
    enn::preset::json::Value root;

    std::ifstream ifs(preset_json_file);

    if (ifs.is_open()) {
        ifs >> root;
    } else {
        ENN_ERR_PRINT("File doesn't exist: %s\n", preset_json_file.c_str());
        return ENN_RET_FAILED;
    }

    /* parsing from json file */
    for (const enn::preset::json::Value &element : root["external_presets"]) {
        std::shared_ptr<PresetConfig> tmp_preset(new PresetConfig());

        if( parse_preset_id(element, tmp_preset) != ENN_RET_SUCCESS)
            return ENN_RET_FAILED;

        if( parse_preset_name(element, tmp_preset) != ENN_RET_SUCCESS)
            return ENN_RET_FAILED;

        ENN_DBG_PRINT(" # PresetId : %d\n", tmp_preset->get_preset_id());
        ENN_DBG_PRINT(" # PresetName : [%s]\n", tmp_preset->get_preset_name().c_str());

        if( parse_preset_vars(element, tmp_preset) != ENN_RET_SUCCESS)
            return ENN_RET_FAILED;

        if( parse_preset_type(element, tmp_preset) != ENN_RET_SUCCESS)
            return ENN_RET_FAILED;

        if( parse_preset_priority(element, tmp_preset) != ENN_RET_SUCCESS)
            return ENN_RET_FAILED;

        tmp_preset->dump();

        preset_config_map_by_id_.insert(make_pair(tmp_preset->get_preset_id(), tmp_preset));
        preset_config_map_by_name_.insert(make_pair(tmp_preset->get_preset_name(), tmp_preset));

    }
    ifs.close();

    return ENN_RET_SUCCESS;
}


uint32_t PresetConfigManager::get_preset_id_by_name(std::string name){
    auto iter = preset_config_map_by_name_.find(name);
    if(iter == preset_config_map_by_name_.end())
        return 0;
    else
        return iter->second->get_preset_id();
}

EnnReturn PresetConfigManager::get_id_from_ext_json(std::string ncpName, std::shared_ptr<uint32_t> id) {
    //TODO(jinhyo01.kim) : parsig preset id from external json file by NCPNAME
    ENN_UNUSED(ncpName);
    ENN_UNUSED(id);
    ENN_DBG_PRINT("get_id_from_ext_json\n");

    return ENN_RET_SUCCESS;
}

EnnReturn PresetConfigManager::get_enn_preset_external_scenario(const uint32_t key, std::shared_ptr<PresetConfig*> preset_config) {
    //TODO(jinhyo01.kim) : parsig preset scenario by key
    ENN_UNUSED(key);
    ENN_UNUSED(preset_config);
    ENN_DBG_PRINT("get_enn_preset_external_scenario\n");

    return ENN_RET_SUCCESS;
}

EnnReturn PresetConfigManager::get_enn_preset_scenario(uint32_t presetId, std::shared_ptr<PresetConfig*> preset_config) {
    //TODO(jinhyo01.kim) : parsig preset scenario by preset id
    ENN_UNUSED(presetId);
    ENN_UNUSED(preset_config);
    ENN_DBG_PRINT("get_enn_preset_scenario\n");

    return ENN_RET_SUCCESS;
}

std::shared_ptr<PresetConfig> PresetConfigManager::get_preset_config_by_id(uint32_t presetId) {
    // TODO(jungho7.kim): add lock for thread-safe data structure
    std::shared_ptr<PresetConfig> ret;
    if (preset_config_map_by_id_.find(presetId) == preset_config_map_by_id_.end()) {
        ENN_WARN_PRINT("fail to get preset_config, Preset ID:%u\n", presetId);
        return nullptr;
    }
    ENN_DBG_PRINT("success to get preset_config, Preset ID:%u\n", presetId);
    ret = preset_config_map_by_id_[presetId];
    return ret;
}

EnnReturn PresetConfigManager::get_preset_id(std::string name, std::shared_ptr<uint32_t> presetId) {
    //TODO(jinhyo01.kim) : parsig preset id by preset name
    ENN_UNUSED(name);
    ENN_UNUSED(presetId);
    ENN_DBG_PRINT("get_preset_id\n");

    return ENN_RET_SUCCESS;
}

EnnReturn PresetConfigManager::delete_enn_preset_scenario() {
    //TODO(jinhyo01.kim) : delete preset scenarios
    ENN_DBG_PRINT("delete_enn_preset_scenario\n");

    return ENN_RET_SUCCESS;
}

};  // namespace preset
};  // namespace enn

