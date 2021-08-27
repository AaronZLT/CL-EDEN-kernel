// enn_api-async_test.cc

#include <algorithm>
#include <atomic>
#include <istream>
#include <future>
#include <mutex>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "enn_api-public.h"
#include "common/enn_utils.h"

const std::string log_prefix = "\033[32m[          ]\033[0m ";
const std::string default_data_path = "/data/vendor/enn/models/pamir/NPU_IV3";
const std::string default_model_file = "NPU_InceptionV3.nnc";
const std::string default_input_files = "NPU_InceptionV3_input_data.bin";
const std::string default_golden_files = "NPU_InceptionV3_golden_data.bin";
const std::string default_session_count = "1";  // TODO: change to some large value if multi-execution is enabled
const std::string default_iteration_count = "10";


static const std::string get_or_default(const char* param, std::string const& def) {
    char *env_str = getenv(param);
    return env_str == nullptr ? def : env_str;
}

static void split_and_push_to(std::string const& src, std::string const& prefix, std::vector<std::string>& dest) {
    std::stringstream ss(src);
    std::istream_iterator<std::string> b(ss);
    std::istream_iterator<std::string> e;
    std::for_each(b, e, [&dest, &prefix](std::string const& s){ dest.push_back(prefix + s); });
}

static void LoadFiles(std::vector<std::string> const& files, std::vector<char*>& buffers, std::vector<size_t>& sizes) {
    char* buf;
    uint32_t size;
    for (auto& file : files) {
        buf = nullptr;
        EXPECT_EQ(
            ENN_RET_SUCCESS,
            enn::util::import_file_to_mem(file.c_str(), &buf, &size, 0)
        ) << "failed to load " << file;
        buffers.push_back(buf);
        sizes.push_back(static_cast<size_t>(size));
    }
}

static void AllocateEnnBuffers(EnnModelId model_id, enn_buf_dir_e direction, uint32_t size, size_t session_count, std::vector<EnnBufferPtr> &buffers) {
    EnnBufferInfo ebi;
    EnnBufferPtr ebp;
    buffers.reserve(session_count * size);
    for (int s = 0; s < session_count; s++) {
        for (int i = 0; i < size; i++) {
            EXPECT_EQ(
                ENN_RET_SUCCESS,
                EnnGetBufferInfoByIndex(model_id, direction, i, &ebi)
            ) << "failed at EnnGetBufferInfoByIndex(" << std::hex << model_id << "," << direction << "," << i << "," << &ebi << ")";
            EXPECT_EQ(
                ENN_RET_SUCCESS,
                EnnCreateBufferCache(ebi.size, &ebp)
            ) << "failed at EnnCreateBufferCache(" << ebi.size << "," << ebp << ")";
            buffers.push_back(ebp);
            EXPECT_EQ(
                ENN_RET_SUCCESS,
                EnnSetBufferByIndexWithSessionId(model_id, direction, i, ebp, s)
            ) << "failed at EnnSetBufferByIndexWithSessionId(" << std::hex << model_id << "," << direction << "," << i << "," << ebp << "," << s << ")";
        }
    }
}

static void FreeBuffers(std::vector<char*>& buffers) {
    for (auto& buf : buffers) {
        free(buf);
    }
    buffers.clear();
}

static void ReleaseEnnBuffers(std::vector<EnnBufferPtr> &buffers) {
        for (auto& ebp : buffers) {
            EXPECT_EQ(
                ENN_RET_SUCCESS,
                EnnReleaseBuffer(ebp)
            ) << "failed at EnnReleaseBuffer(" << ebp << ")";
        }
        buffers.clear();
}

class ENN_API_ASYNC_TEST : public ::testing::Test {
protected:
    void SetUp() override {
        PrepareParams();
        EXPECT_EQ(
            ENN_RET_SUCCESS,
            EnnInitialize()
        ) << "failed at EnnInitialize()";;
        OpenModel();
    }

    void TearDown() override {
        ReleaseBuffers();
        CloseModel();
        EXPECT_EQ(
            ENN_RET_SUCCESS,
            EnnDeinitialize()
        ) << "failed at EnnDeinitialize()";
    }

    void PrintNote() {
        std::cout << log_prefix << "Note: set envirement variables to change model, input and golden fiels." << std::endl
                  << log_prefix << " export ENN_DATA_PATH=path_to_data" << std::endl
                  << log_prefix << " export ENN_MODEL=model_file_name" << std::endl
                  << log_prefix << " export ENN_INPUT=\"input_file_1 input_file_2 ...\"" << std::endl
                  << log_prefix << " export ENN_GOLDEN=\"golden_file_1 golden_file_2 ...\"" << std::endl
                  << log_prefix << " export ENN_SESSIONS=number_of_sessions" << std::endl
                  << log_prefix << " export ENN_ITER=number_of_iterations" << std::endl;
    }

    void PrintParams() {
        std::cout << log_prefix << "model: " << model_file_ << std::endl;
        std::cout << log_prefix << "input: ";
        std::copy(input_file_list_.begin(), input_file_list_.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << std::endl;
        std::cout << log_prefix << "golden: ";
        std::copy(golden_file_list_.begin(), golden_file_list_.end(), std::ostream_iterator<std::string>(std::cout, " "));
        std::cout << std::endl;
        std::cout << log_prefix << "session count: " << session_count_ << std::endl;
        std::cout << log_prefix << "iteration count: " << iteration_count_ << std::endl;
    }

    EnnModelId GetModelId() {
        return model_id_;
    }

    size_t GetSessionCount() {
        return session_count_;
    }

    size_t GetIterationCount() {
        return iteration_count_;
    }

    void PrepareToExecute() {
        LoadInputAndGoldenFiles();
        AllocateBuffers();
    }

    void FillInputBuffers(int session) {
        size_t size = input_buffers_.size();
        for (size_t i = 0; i < size; i++) {
            std::memcpy(enn_in_buffers_[size * session + i]->va,
                reinterpret_cast<void*>(input_buffers_[i]),
                std::min(static_cast<size_t>(enn_in_buffers_[size * session + i]->size), input_buffer_sizes_[i]));
        }
    }

    bool IsMatchingWithGolden(int session) {
        size_t size = golden_buffers_.size();
        for (size_t i = 0; i < size; i++) {
            // TODO: change to acquire UnitType from env
            auto ret =  enn::util::CompareBuffersWithThreshold<float>(
                            golden_buffers_[i],
                            reinterpret_cast<char*>(enn_out_buffers_[size * session + i]->va),
                            golden_buffer_sizes_[i], nullptr, enn::util::BUFFER_COMPARE_THRESHOLD, false);
            if (ret) return false;
        }
        return true;
    }

private:
    void PrepareParams() {
        std::string data_path = get_or_default("ENN_DATA_PATH", default_data_path);
        if (data_path.back() != '/') data_path.push_back('/');
        model_file_ = data_path + get_or_default("ENN_MODEL", default_model_file);
        split_and_push_to(get_or_default("ENN_INPUT", default_input_files), data_path, input_file_list_);
        split_and_push_to(get_or_default("ENN_GOLDEN", default_golden_files), data_path, golden_file_list_);
        session_count_ = std::stoul(get_or_default("ENN_SESSIONS", default_session_count));
        iteration_count_ = std::stoul(get_or_default("ENN_ITER", default_iteration_count));
    }

    void OpenModel() {
        EXPECT_EQ(
            ENN_RET_SUCCESS,
            EnnOpenModel(model_file_.c_str(), &model_id_)
        ) << "failed at EnnOpenModel(" << model_file_ << ",...)";
    }

    void CloseModel() {
        EXPECT_EQ(
            ENN_RET_SUCCESS,
            EnnCloseModel(model_id_)
        ) << "failed at EnnCloseModel(" << std::hex << model_id_ << ")";
    }

    void LoadInputAndGoldenFiles() {
        input_buffers_.reserve(input_file_list_.size());
        input_buffer_sizes_.reserve(input_file_list_.size());
        LoadFiles(input_file_list_, input_buffers_, input_buffer_sizes_);
        golden_buffers_.reserve(golden_file_list_.size());
        golden_buffer_sizes_.reserve(golden_file_list_.size());
        LoadFiles(golden_file_list_, golden_buffers_, golden_buffer_sizes_);
    }

    void AllocateBuffers() {
        NumberOfBuffersInfo nobi;
        EXPECT_EQ(
            ENN_RET_SUCCESS,
            EnnGetBuffersInfo(model_id_, &nobi)
        ) << "failed at EnnGetBuffersInfo(" << std::hex << model_id_ << ", ..)";
        EXPECT_EQ(nobi.n_in_buf, input_buffers_.size());
        EXPECT_EQ(nobi.n_out_buf, golden_buffers_.size());

        EXPECT_EQ(
            ENN_RET_SUCCESS,
            EnnGenerateBufferSpace(model_id_, session_count_)
        ) << "failed at EnnGenerateBufferSpace(" << std::hex << model_id_ << "," << session_count_ << ")";

        AllocateEnnBuffers(model_id_, ENN_DIR_IN, nobi.n_in_buf, session_count_, enn_in_buffers_);
        AllocateEnnBuffers(model_id_, ENN_DIR_OUT, nobi.n_out_buf, session_count_, enn_out_buffers_);

        for (int s = 0; s < session_count_; s++) {
            EXPECT_EQ(
                ENN_RET_SUCCESS,
                EnnBufferCommitWithSessionId(model_id_, s)
            ) << "failed at EnnBufferCommitWithSessionId(" << std::hex << model_id_ << "," << s << ")";
        }
    }

    void ReleaseBuffers() {
        FreeBuffers(golden_buffers_);
        FreeBuffers(input_buffers_);
        ReleaseEnnBuffers(enn_in_buffers_);
        ReleaseEnnBuffers(enn_out_buffers_);
    }

    EnnModelId model_id_;
    std::vector<EnnBufferPtr> enn_in_buffers_;
    std::vector<EnnBufferPtr> enn_out_buffers_;

    std::string model_file_;
    std::vector<std::string> input_file_list_;
    std::vector<std::string> golden_file_list_;
    size_t session_count_;
    size_t iteration_count_;

    std::vector<char*> input_buffers_;
    std::vector<size_t> input_buffer_sizes_;
    std::vector<char*> golden_buffers_;
    std::vector<size_t> golden_buffer_sizes_;

};


TEST_F(ENN_API_ASYNC_TEST, _notice) {
    PrintParams();
    PrintNote();
}

TEST_F(ENN_API_ASYNC_TEST, invoke_single_execution) {
    PrepareToExecute();
    const EnnModelId model_id = GetModelId();
    const size_t iteration_count = GetIterationCount();
    // const size_t session_count = GetSessionCount();
    for (size_t i = 0; i < iteration_count; i++) {
        FillInputBuffers(0);
        EXPECT_EQ(
            ENN_RET_SUCCESS,
            EnnExecuteModelAsync(model_id)
        ) << "failed at EnnExecuteModelAsync(" << std::hex << model_id << ")";
        EXPECT_EQ(
            ENN_RET_SUCCESS,
            EnnExecuteModelWait(model_id)
        ) << "failed at EnnExecuteModelWait(" << std::hex << model_id << ")";
        EXPECT_TRUE(
            IsMatchingWithGolden(0)
        ) << "failed at IsMatchingWithGolden(0)";
    }
}

TEST_F(ENN_API_ASYNC_TEST, invoke_single_execution_with_rotating_session_ids) {
    PrepareToExecute();
    const EnnModelId model_id = GetModelId();
    const size_t iteration_count = GetIterationCount();
    const size_t session_count = GetSessionCount();
    for (size_t i = 0; i < iteration_count; i++) {
        int session_id = i % session_count;
        FillInputBuffers(session_id);
        EXPECT_EQ(
            ENN_RET_SUCCESS,
            EnnExecuteModelWithSessionIdAsync(model_id, session_id)
        ) << "failed at EnnExecuteModelWithSessionIdAsync(" << std::hex << model_id << "," << session_id << ")";
        EXPECT_EQ(
            ENN_RET_SUCCESS,
            EnnExecuteModelWithSessionIdWait(model_id, session_id)
        ) << "failed at EnnExecuteModelWithSessionIdWait(" << std::hex << model_id << "," << session_id << ")";
        EXPECT_TRUE(
            IsMatchingWithGolden(session_id)
        ) << "failed at IsMatchingWithGolden(0)";
    }
}

TEST_F(ENN_API_ASYNC_TEST, invoke_sessions_in_discrete_threads) {
    PrepareToExecute();
    const EnnModelId model_id = GetModelId();
    std::atomic<int> iter_counter = GetIterationCount();
    const size_t session_count = GetSessionCount();
    std::thread ths[session_count];
    for (size_t s = 0; s < session_count; s++) {
        ths[s] = std::thread([&]() {
            const int session_id = static_cast<int>(s - 1);
            while (0 <= iter_counter.fetch_sub(1)) {
                FillInputBuffers(session_id);
                // effectivly same with invoking EnnExecuteModelWithSessionId(int)
                // in threads as much as number of sessions
                ENN_DBG_PRINT("Start to Execute model asynchronously with model(0x%" PRIX64 ") and session(0x%X)\n",
                                                                                model_id, session_id);
                EXPECT_EQ(
                    ENN_RET_SUCCESS,
                    EnnExecuteModelWithSessionIdAsync(model_id, session_id)
                ) << "failed at EnnExecuteModelWithSessionIdAsync(" << std::hex << model_id << "," << session_id << ")";
                ENN_DBG_PRINT("Finish to Execute model asynchronously with model(0x%" PRIX64 ") and session(0x%X)\n",
                                                                                model_id, session_id);
                ENN_DBG_PRINT("Star to wait done to execute asynchronously with model(0x%" PRIX64 ") and session(0x%X)\n",
                                                                                model_id, session_id);
                EXPECT_EQ(
                    ENN_RET_SUCCESS,
                    EnnExecuteModelWithSessionIdWait(model_id, session_id)
                ) << "failed at EnnExecuteModelWithSessionIdWait(" << std::hex << model_id << "," << session_id << ")";
                ENN_DBG_PRINT("Finish to wait done to execute asynchronously with model(0x%" PRIX64 ") and session(0x%X)\n",
                                                                                model_id, session_id);
                EXPECT_TRUE(
                    IsMatchingWithGolden(session_id)
                ) << "failed at IsMatchingWithGolden(" << session_id << ")";
            }
        });
    }
    for (auto& th : ths) {
        th.join();
    }
}

TEST_F(ENN_API_ASYNC_TEST, invoke_sessions_as_produder_consumer_pattern) {
    PrepareToExecute();
    const EnnModelId model_id = GetModelId();
    const size_t iteration_count = GetIterationCount();
    const size_t session_count = GetSessionCount();

    std::vector<int> sids_idle, sids_busy;
    sids_idle.reserve(session_count);
    sids_busy.reserve(session_count);
    for (int i = 0; i < session_count; i++) {
        sids_idle.push_back(i);
    }

    std::mutex m_idle, m_busy;
    std::condition_variable cv_idle, cv_busy;

    auto f1 = std::async(std::launch::async, [&]() {
        int counter = 0;
        while (counter++ < iteration_count) {
            int session_id;
            {
                std::unique_lock<std::mutex> lock(m_idle);
                while (sids_idle.empty()) cv_idle.wait(lock);
                session_id = sids_idle.front(); sids_idle.erase(sids_idle.begin());
            }
            FillInputBuffers(session_id);
            ENN_TST_PRINT("   -ExecuteAsync [[ session=%d counter=%d...\n", session_id, counter);
            EXPECT_EQ(
                ENN_RET_SUCCESS,
                EnnExecuteModelWithSessionIdAsync(model_id, session_id)
            ) << "failed at EnnExecuteModelWithSessionIdAsync(" << std::hex << model_id << "," << session_id << ")";
            {
                std::unique_lock<std::mutex> lock(m_busy);
                sids_busy.push_back(session_id);
                cv_busy.notify_all();
            }
        }
        ENN_TST_PRINT("   -ExecuteAsync done executing for %d times\n", counter - 1);
    });

    auto f2 = std::async(std::launch::async, [&]() {
        int counter = 0;
        while (counter++ < iteration_count) {
            int session_id;
            {
                std::unique_lock<std::mutex> lock(m_busy);
                while (sids_busy.empty()) cv_busy.wait(lock);
                session_id = sids_busy.front(); sids_busy.erase(sids_busy.begin());
            }
            ENN_TST_PRINT("   -ExecuteAsync -- session=%d counter=%d...\n", session_id, counter);
            EXPECT_EQ(
                ENN_RET_SUCCESS,
                EnnExecuteModelWithSessionIdWait(model_id, session_id)
            ) << "failed at EnnExecuteModelWithSessionIdWait(" << std::hex << model_id << "," << session_id << ")";
            ENN_TST_PRINT("   -ExecuteAsync ]] session=%d counter=%d...\n", session_id, counter);
            EXPECT_TRUE(
                IsMatchingWithGolden(session_id)
            ) << "failed at IsMatchingWithGolden(" << session_id << ")";
            {
                std::unique_lock<std::mutex> lock(m_idle);
                sids_idle.push_back(session_id);
                cv_idle.notify_all();
            }
        }
        ENN_TST_PRINT("   -ExecuteAsync done waiting for %d times\n", counter - 1);
    });

    f1.wait();
    f2.wait();

}

TEST_F(ENN_API_ASYNC_TEST, attempt_execution_before_preparing_buffers) {
    EXPECT_NE(
        ENN_RET_SUCCESS,
        EnnExecuteModelAsync(GetModelId())
    ) << "failed at EnnExecuteModelAsync(" << std::hex << GetModelId() << ")";
}

TEST_F(ENN_API_ASYNC_TEST, attempt_execution_with_invalid_model_id) {
    const EnnModelId model_ids[] = {
        0x0,
        std::numeric_limits<uint64_t>::max()
    };
    for (auto& model_id: model_ids) {
        EXPECT_NE(
            ENN_RET_SUCCESS,
            EnnExecuteModelAsync(model_id)
        ) << "failed at EnnExecuteModelAsync(" << std::hex << model_id << ")";
    }
}

TEST_F(ENN_API_ASYNC_TEST, attempt_execution_with_invalid_session_id) {
    PrepareToExecute();
    int session_ids[] = {
        0,
        -1,
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()
    };
    session_ids[0] = GetSessionCount() + 1;  // replace 0 with invalid session id
    for (auto& session_id: session_ids) {
        EXPECT_NE(
            ENN_RET_SUCCESS,
            EnnExecuteModelWithSessionIdAsync(GetModelId(), session_id)
        ) << "failed at EnnExecuteModelWithSessionIdAsync(" << std::hex << GetModelId() << "," << session_id << ")";
    }
}

TEST_F(ENN_API_ASYNC_TEST, attempt_waiting_with_invalid_model_id) {
    const EnnModelId model_ids[] = {
        0x0,
        std::numeric_limits<uint64_t>::max()
    };
    for (auto& model_id: model_ids) {
        EXPECT_NE(
            ENN_RET_SUCCESS,
            EnnExecuteModelWait(model_id)
        ) << "failed at EnnExecuteModelWait(" << std::hex << model_id << ")";
    }
}

TEST_F(ENN_API_ASYNC_TEST, attempt_waiting_with_invalid_session_id) {
    PrepareToExecute();
    int session_ids[] = {
        0,
        -1,
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()
    };
    session_ids[0] = GetSessionCount() + 1;  // replace 0 with invalid session id
    for (auto& session_id: session_ids) {
        EXPECT_NE(
            ENN_RET_SUCCESS,
            EnnExecuteModelWithSessionIdWait(GetModelId(), session_id)
        ) << "failed at EnnExecuteModelWithSessionIdWait(" << std::hex << GetModelId() << "," << session_id << ")";
    }
}
