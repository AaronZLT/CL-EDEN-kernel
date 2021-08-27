/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */


#ifndef ENN_TEST_H_
#define ENN_TEST_H_

#include <string>
#include <vector>

#include "enn_test_type.hpp"
#include "enn_test_reporter.h"

#include "client/enn_api.h"
#include "tool/profiler/include/ExynosNnProfilerApi.h"

namespace enn_test {
class EnnTest {
    TestParams test_params;
    std::vector<std::vector<TestReporter*>> test_reporter;  // [thread][session]
    std::string reportPath;

    void Initialize();
    void SetPreference();
    void OpenModel(EnnModelId &model_id);
    void AllocateBuffers(const EnnModelId model_id, TestBuffers& test_buffer);
    void LoadInput(TestBuffers& test_buffer);
    void LoadOutput(TestBuffers& test_buffer);
    void PrepareToExecute(const EnnModelId model_id);
    void Execute(const EnnModelId model_id, const int32_t session_id);
    void Execute_async(const EnnModelId model_id, const int32_t session_id);
    EnnTestReturn CompareGolden(TestBuffers& test_buffer, const int32_t iter,
                                const int32_t session_id);
    EnnTestReturn Execute_iter(const EnnModelId model_id, TestBuffers& test_buffer,
                               const int32_t thread_id = 0);
    EnnTestReturn Execute_duration(const EnnModelId model_id, TestBuffers& test_buffer);
    void DumpOutput(TestBuffers& test_buffer);
    void ReleaseBuffer(TestBuffers& test_buffer);
    void CloseModel(const EnnModelId model_id);
    void Deinit();

    void CreateReporters();
    void DeleteReporters();

    void ValidateNumOfBuffers(const TestBuffers& test_buffer) const;

  public:
    EnnTest(TestParams& test_params_) : test_params(test_params_) {
        CreateReporters();
    }

    ~EnnTest() {
        DeleteReporters();
    }

    EnnTestReturn run();
};
}  // namespace enn_test
#endif  // ENN_TEST_H_
