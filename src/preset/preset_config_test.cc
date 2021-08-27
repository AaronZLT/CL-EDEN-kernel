#include <iostream>
#include <fstream>
#include <string>

#include "preset_config.hpp"
#include "gtest/gtest.h"

namespace enn {
namespace test {
namespace internal {

class ENN_GT_PRESET_CONFIG_TEST : public testing::Test {

 public:
    ENN_GT_PRESET_CONFIG_TEST(){}

    ~ENN_GT_PRESET_CONFIG_TEST(){}

 protected:

    //TODO (jinhyo01.kim) : I will add these function when there is initialize setup job or deinitialize cleanning job
    /*
    void SetUp() override { }

    void TearDown() override { }
    */
};

TEST_F(ENN_GT_PRESET_CONFIG_TEST, parsing_preset_json_file) {

    auto preset_manager = enn::preset::PresetConfigManager::get_instance();

    //TODO (jinhyo01.kim) : move to SetUp() function
    const std::string json_file = "/vendor/etc/eden/enn_preset.json";

    uint32_t test_result = 0;
    ENN_INFO_PRINT("parsing preset : %d\n",test_result);
    ENN_INFO_PRINT("json file : %s\n", json_file.c_str());

    EXPECT_EQ(ENN_RET_SUCCESS, preset_manager->parse_scenario_from_ext_json(json_file));
}

TEST_F(ENN_GT_PRESET_CONFIG_TEST, parsing_preset_json_file_v2) {

    auto preset_manager = enn::preset::PresetConfigManager::get_instance();

    //TODO (jinhyo01.kim) : move to SetUp() function
    const std::string json_file = "/vendor/etc/eden/enn_preset_v2.json";

    uint32_t test_result = 0;
    ENN_INFO_PRINT("parsing preset : %d\n",test_result);
    ENN_INFO_PRINT("json file : %s\n", json_file.c_str());

    EXPECT_EQ(ENN_RET_SUCCESS, preset_manager->parse_scenario_from_ext_json_v2(json_file));
}

}  // namespace internal
}  // namespace test
}  // namespace enn