/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

#ifndef SRC_COMMON_ENN_PREFERENCE_GENERATOR_H_
#define SRC_COMMON_ENN_PREFERENCE_GENERATOR_H_

#include "common/enn_common_type.h"
#include <mutex>
#include <vector>

namespace enn {
namespace preference {

class EnnPreferenceGenerator {
  public:
    EnnPreferenceGenerator() {
        reset_as_default();
    }

    EnnPreferenceGenerator(std::vector<uint32_t> pref) {
        import_preference_from_vector(pref);
    }

    virtual ~EnnPreferenceGenerator() {}

    EnnReturn reset_as_default() {
        preference = default_preference;
        return ENN_RET_SUCCESS;
    }  // set default

    EnnReturn set_preset_id(uint32_t t) {
        preference.preset_id = t;
        return ENN_RET_SUCCESS;
    }

    EnnReturn set_pref_mode(uint32_t t) {
        preference.pref_mode = t;
        return ENN_RET_SUCCESS;
    }

    uint32_t get_preset_id() {
        return preference.preset_id;
    }

    uint32_t get_pref_mode() {
        return preference.pref_mode;
    }

    uint32_t get_target_latency() {
        return preference.target_latency;
    }

    uint32_t get_tile_num() {
        return preference.tile_num;
    }

    uint32_t get_core_affinity() {
        return preference.core_affinity;
    }

    uint32_t get_priority() {
        return preference.priority;
    }

    uint32_t get_custom_0() {
        return preference.custom[0];
    }

    uint32_t get_custom_1() {
        return preference.custom[1];
    }

    EnnReturn set_target_latency(uint32_t t) {
        preference.target_latency = t;
        return ENN_RET_SUCCESS;
    }

    EnnReturn set_tile_num(uint32_t t) {
        preference.tile_num = t;
        return ENN_RET_SUCCESS;
    }

    EnnReturn set_core_affinity(uint32_t t) {
        preference.core_affinity = t;
        return ENN_RET_SUCCESS;
    }

    EnnReturn set_priority(uint32_t t) {
        preference.priority = t;
        return ENN_RET_SUCCESS;
    }

    EnnReturn set_custom_0(uint32_t t) {
        preference.custom[0] = t;
        return ENN_RET_SUCCESS;
    }

    EnnReturn set_custom_1(uint32_t t) {
        preference.custom[1] = t;
        return ENN_RET_SUCCESS;
    }

    uint32_t get_stream_size() {
        return static_cast<uint32_t>(sizeof(preference) / sizeof(uint32_t));
    }

    uint32_t * get_stream_pointer() {
        return reinterpret_cast<uint32_t *>(&preference);
    }

    std::vector<uint32_t> export_preference_to_vector() {
        std::vector<uint32_t> ret(get_stream_pointer(), get_stream_pointer() + get_stream_size());
        return ret;
    }

    EnnReturn import_preference_from_vector(std::vector<uint32_t> pref) {
        if (static_cast<uint32_t>(pref.size()) > get_stream_size()) {
            ENN_ERR_COUT << "Import preference is bigger than own size :" << get_stream_size();
            return ENN_RET_FAILED;
        }
        auto ptr = get_stream_pointer();
        for (auto i = 0; i < pref.size(); i++)
            ptr[i] = pref[i];
        return ENN_RET_SUCCESS;
    }

    void show() {
        uint32_t *ptr = get_stream_pointer();
        for (auto i = 0; i < get_stream_size(); i++) {
            ENN_DBG_PRINT(" # Preset [%d]: 0x%X\n", i, ptr[i]);
        }
    }

  private:
    std::mutex pref_gen_mutex;
    EnnPreference preference;
    const EnnPreference default_preference = {0, ENN_PREF_MODE_BOOST_ON_EXE, 0, 1, 0xFFFFFFFF, 0, {0, 0}};
};


}
}

#endif  // SRC_COMMON_ENN_PREFERENCE_GENERATOR_H_
