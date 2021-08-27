/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <vector>
#include <future>       // async
#include <unistd.h>     // sleep

#include "enn_test.h"
#include "enn_test_log.h"

namespace enn_test {

void EnnTest::CreateReporters() {
    test_reporter.resize(test_params.thread_num);

    for (int thread_id = 0; thread_id < test_params.thread_num; ++thread_id) {
        for (int session_id = 0; session_id < test_params.session_num; ++session_id) {
            int32_t reporter_id = thread_id * 1000 + session_id;

            if (test_params.duration > 0) {
               test_reporter[thread_id].push_back(new TestReporter());
            } else if (test_params.reportPath.empty()) {
                test_reporter[thread_id].push_back(new TestReporter(reporter_id,
                                                                    test_params.repeat,
                                                                    test_params.iter));
            } else {
                test_reporter[thread_id].push_back(new TestReporter(reporter_id,
                                                                    test_params.repeat,
                                                                    test_params.iter,
                                                                    test_params.reportPath));
            }
        }
    }
}

void EnnTest::DeleteReporters() {
    for (int t = 0; t < test_reporter.size(); ++t) {
        for (int s = 0; s < test_reporter[t].size(); ++s) {
            delete test_reporter[t][s];
        }
    }
}

EnnTestReturn EnnTest::run() {
    ENN_TEST_DEBUG("+");
    START_PROFILER(123456788);
    const int32_t default_thread_id = 0;
    EnnTestReturn ret = RET_SUCCESS;

    try {
        std::vector<TestBuffers> test_buffers(test_params.thread_num);
        std::vector<EnnModelId> model_ids(test_params.thread_num);

        Initialize();
        SetPreference();

        for (int thread_id = 0; thread_id < test_params.thread_num; ++thread_id) {
            OpenModel(model_ids[thread_id]);

            AllocateBuffers(model_ids[thread_id], test_buffers[thread_id]);

            LoadInput(test_buffers[thread_id]);
            LoadOutput(test_buffers[thread_id]);

            PrepareToExecute(model_ids[thread_id]);
        }

        if (test_params.delay > 0) {
            sleep(test_params.delay);
        }

        if (test_params.thread_num == 1) {
            if (test_params.duration == 0) {
                ret = Execute_iter(model_ids[default_thread_id], test_buffers[default_thread_id]);
            } else {
                ret = Execute_duration(model_ids[default_thread_id], test_buffers[default_thread_id]);
            }
        } else if (test_params.thread_num > 1) {
            std::vector<std::future<void>> futures(test_params.thread_num);
            for (int thread_id = 0; thread_id < test_params.thread_num; ++thread_id) {
                futures[thread_id] = std::async([&, thread_id]() {
                    if (test_params.duration == 0) {
                        ret = Execute_iter(model_ids[thread_id], test_buffers[thread_id], thread_id);
                    } else {
                        ret = Execute_duration(model_ids[thread_id], test_buffers[thread_id]);
                    }
                });
            }

            for(auto& future : futures) {
                future.get();
            }
        } else {
            ENN_TEST_ERR("Thread_num error : %d\n", test_params.thread_num);
            throw RET_INVALID_PARAM;
        }

        if (test_params.dump_output) {
            DumpOutput(test_buffers[default_thread_id]);
        }

        for (int thread_id = 0; thread_id < test_params.thread_num; ++thread_id) {
            ReleaseBuffer(test_buffers[thread_id]);

            CloseModel(model_ids[thread_id]);
        }

        Deinit();
    } catch (EnnTestReturn err) {
        // Todo : release buffer
        FINISH_PROFILER(123456788);
        return err;
    }

    FINISH_PROFILER(123456788);
    ENN_TEST_DEBUG("-");
    return ret;
}

}  // namespace enn_test
