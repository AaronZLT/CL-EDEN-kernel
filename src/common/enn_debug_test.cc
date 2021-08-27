#include <iostream>
#include <climits>
#include <cstdlib>
#include <map>
#include <mutex>
#include <memory>
#include <vector>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>

#include "gtest/gtest.h"
#include "common/enn_debug.h"
#include "common/enn_utils.h"

#ifndef __ANDROID__
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
#endif

namespace enn {
namespace test {
namespace internal {

class ENN_GT_API_PRINT_TEST : public testing::Test {
protected:
    virtual void SetUp() {
        bkup_mask = enn::debug::DbgPrintManager::GetInstance().get_print_mask();
    }

    virtual void TearDown() {
        enn::debug::DbgPrintManager::GetInstance().set_mask(bkup_mask);
    }

private:
    int bkup_mask;
};

/* Because Android uses logcat, the test cannot capture console output */
#if !defined(ENN_BUILD_RELEASE) && !defined(__ANDROID__)
struct EnnTstStrFeed {
    enn::debug::DbgPartition zone;
    std::string value;
};

const std::vector<EnnTstStrFeed> tstFeed = {
    {enn::debug::DbgPartition::kError, "kError test"},   {enn::debug::DbgPartition::kWarning, "kWarning test"},
    {enn::debug::DbgPartition::kInfo, "kInfo test"},     {enn::debug::DbgPartition::kDebug, "kDebug test"},
    {enn::debug::DbgPartition::kMemory, "kMemory test"}, {enn::debug::DbgPartition::kTest, "kTest test"},
    {enn::debug::DbgPartition::kUser, "kUser test"},
};

int test_start_capture(const char* filename) {
    static int savestdout = 0;
    savestdout = dup(savestdout != 0 ? savestdout : STDOUT_FILENO);
    freopen(filename, "w", stdout);

    return savestdout;
}

int test_stop_capture(int my_stdout) {
    fflush(stdout);
    fclose(stdout);
    auto myfp = fdopen(my_stdout, "w");
    *stdout = *myfp;

    return 0;
}

bool verify_capture(const char* filename, uint32_t mask) {
    int index = 0;
    std::ifstream fin(filename);
    if (fin.is_open()) {
        std::string line;
        while (std::getline(fin, line)) {
            /* pass if each zone is hide */
            while (!(mask & ZONE_BIT(static_cast<int>(tstFeed[index].zone)))) index++;
            if (line.rfind(tstFeed[index].value) == std::string::npos)
                return false;
            index++;
        }
    } else {
        std::cout << "Open Error" << std::endl;
        return false;
    }
    return true;
}

void show_test() {
    for (auto& feed : tstFeed) {
        switch (feed.zone) {
            case enn::debug::DbgPartition::kError:
                ENN_ERR_PRINT("%s\n", feed.value.c_str());
                break;
            case enn::debug::DbgPartition::kWarning:
                ENN_WARN_PRINT("%s\n", feed.value.c_str());
                break;
            case enn::debug::DbgPartition::kInfo:
                ENN_INFO_PRINT("%s\n", feed.value.c_str());
                break;
            case enn::debug::DbgPartition::kDebug:
                ENN_DBG_PRINT("%s\n", feed.value.c_str());
                break;
            case enn::debug::DbgPartition::kMemory:
                ENN_MEM_PRINT("%s\n", feed.value.c_str());
                break;
            case enn::debug::DbgPartition::kTest:
                ENN_TST_PRINT("%s\n", feed.value.c_str());
                break;
            case enn::debug::DbgPartition::kUser:
                ENN_USER_PRINT("%s\n", feed.value.c_str());
                break;
        }
    }
}

void show_test_stream() {
    for (auto& feed : tstFeed) {
        switch (feed.zone) {
            case enn::debug::DbgPartition::kError:
                ENN_ERR_COUT << feed.value << std::endl;
                break;
            case enn::debug::DbgPartition::kWarning:
                ENN_WARN_COUT << feed.value << std::endl;
                break;
            case enn::debug::DbgPartition::kInfo:
                ENN_INFO_COUT << feed.value << std::endl;
                break;
            case enn::debug::DbgPartition::kDebug:
                ENN_DBG_COUT << feed.value << std::endl;
                break;
            case enn::debug::DbgPartition::kMemory:
                ENN_MEM_COUT << feed.value << std::endl;
                break;
            case enn::debug::DbgPartition::kTest:
                ENN_TST_COUT << feed.value << std::endl;
                break;
            case enn::debug::DbgPartition::kUser:
                ENN_USER_COUT << feed.value << std::endl;
                break;
        }
    }
}

/* in Release mode, the system forcibly restrict print, so the following 2 tests are not working */
/* in Android, the logcat is not write on the terminal (write on logcat pages) */
TEST_F(ENN_GT_API_PRINT_TEST, check_all_debug_zone) {
    const char* filename = "test_redirect.txt";
    const uint32_t mask = 0xFFFFFFFF;

    /* set all mesages are shown */
    auto ret_fp = test_start_capture(filename);  // redirect stdout to file
    show_test();
    test_stop_capture(ret_fp);
    EXPECT_EQ(verify_capture(filename, mask), true);
}

TEST_F(ENN_GT_API_PRINT_TEST, check_mask_set_0x13) {
    constexpr const char* filename = "test_redirect.txt";
    constexpr uint32_t mask = 0x13;

    /* set bit 0, 1, 4(err, warning, test) messages are shown */
    enn::debug::DbgPrintManager::GetInstance().set_mask(mask);
    auto ret_fp = test_start_capture(filename);
    show_test();
    test_stop_capture(ret_fp);
    EXPECT_EQ(verify_capture(filename, mask), true);
}

TEST_F(ENN_GT_API_PRINT_TEST, check_set_debug_zone_and_check_output) {
    constexpr const char* filename = "test_redirect.txt";
    constexpr uint32_t mask = 0;

    enn::debug::DbgPrintManager::GetInstance().set_mask(mask);

    enn::debug::DbgPrintManager::GetInstance().enn_set_debug_zone(enn::debug::DbgPartition::kError);
    auto my_mask = enn::debug::DbgPrintManager::GetInstance().get_print_mask();
    EXPECT_EQ(my_mask, ZONE_BIT_MASK(enn::debug::DbgPartition::kError));
    auto ret_fp = test_start_capture(filename);
    show_test();
    test_stop_capture(ret_fp);
    EXPECT_EQ(verify_capture(filename, my_mask), true);
}

TEST_F(ENN_GT_API_PRINT_TEST, check_all_debug_zone_stream) {
    const char* filename = "test_redirect.txt";
    const uint32_t mask = 0xFFFFFFFF;

    /* set all mesages are shown */
    auto ret_fp = test_start_capture(filename);  // redirect stdout to file
    show_test_stream();
    test_stop_capture(ret_fp);
    EXPECT_EQ(verify_capture(filename, mask), true);
}

TEST_F(ENN_GT_API_PRINT_TEST, check_mask_set_0x13_stream) {
    constexpr const char* filename = "test_redirect.txt";
    constexpr uint32_t mask = 0x13;

    /* set bit 0, 1, 4(err, warning, test) messages are shown */
    enn::debug::DbgPrintManager::GetInstance().set_mask(mask);
    auto ret_fp = test_start_capture(filename);
    show_test_stream();
    test_stop_capture(ret_fp);
    EXPECT_EQ(verify_capture(filename, mask), true);
}

TEST_F(ENN_GT_API_PRINT_TEST, check_set_debug_zone_and_check_output_stream) {
    constexpr const char* filename = "test_redirect.txt";
    constexpr uint32_t mask = 0;

    enn::debug::DbgPrintManager::GetInstance().set_mask(mask);

    enn::debug::DbgPrintManager::GetInstance().enn_set_debug_zone(enn::debug::DbgPartition::kError);
    auto my_mask = enn::debug::DbgPrintManager::GetInstance().get_print_mask();
    EXPECT_EQ(my_mask, ZONE_BIT_MASK(enn::debug::DbgPartition::kError));
    auto ret_fp = test_start_capture(filename);
    show_test_stream();
    test_stop_capture(ret_fp);
    EXPECT_EQ(verify_capture(filename, my_mask), true);
}

#endif


TEST_F(ENN_GT_API_PRINT_TEST, test_with_log_tag) {
    ENN_INFO_PRINT_TAG("TEST_TAG", "%s %d\n", "hello, world!", 2324);
}

TEST_F(ENN_GT_API_PRINT_TEST, check_set_debug_zone_is_working) {
    uint32_t mask = 0;
    enn::debug::DbgPrintManager::GetInstance().set_mask(mask);

    /* turn on warning message */
    EXPECT_EQ(enn::debug::DbgPrintManager::GetInstance().enn_set_debug_zone(enn::debug::DbgPartition::kWarning),
              ENN_RET_SUCCESS);

    /* get mask */
    auto my_mask = enn::debug::DbgPrintManager::GetInstance().get_print_mask();

    /* check mask is correctly set */
	EXPECT_EQ(my_mask, ZONE_BIT_MASK(enn::debug::DbgPartition::kWarning));
}

TEST_F(ENN_GT_API_PRINT_TEST, check_clr_debug_zone_is_working) {
    uint32_t mask = ZONE_BIT_MASK(enn::debug::DbgPartition::kWarning);

    enn::debug::DbgPrintManager::GetInstance().set_mask(mask);

    /* turn on warning message */
    EXPECT_EQ(enn::debug::DbgPrintManager::GetInstance().enn_clr_debug_zone(enn::debug::DbgPartition::kWarning),
              ENN_RET_SUCCESS);

    /* get mask */
    auto my_mask = enn::debug::DbgPrintManager::GetInstance().get_print_mask();

    /* check mask is correctly set */
    EXPECT_EQ(my_mask, 0);
}

TEST_F(ENN_GT_API_PRINT_TEST, check_set_clr_debug_zone_are_working_complex_test) {
    uint32_t mask = 0;

    enn::debug::DbgPrintManager::GetInstance().set_mask(mask);

    /* turn on warning message */
    EXPECT_EQ(enn::debug::DbgPrintManager::GetInstance().enn_set_debug_zone(enn::debug::DbgPartition::kWarning),
              ENN_RET_SUCCESS);
    EXPECT_EQ(enn::debug::DbgPrintManager::GetInstance().enn_set_debug_zone(enn::debug::DbgPartition::kError),
              ENN_RET_SUCCESS);
    EXPECT_EQ(enn::debug::DbgPrintManager::GetInstance().get_print_mask(),
              ZONE_BIT_MASK(enn::debug::DbgPartition::kWarning) | ZONE_BIT_MASK(enn::debug::DbgPartition::kError));
    EXPECT_EQ(enn::debug::DbgPrintManager::GetInstance().enn_clr_debug_zone(enn::debug::DbgPartition::kWarning),
              ENN_RET_SUCCESS);
    EXPECT_EQ(enn::debug::DbgPrintManager::GetInstance().get_print_mask(), ZONE_BIT_MASK(enn::debug::DbgPartition::kError));
    EXPECT_EQ(enn::debug::DbgPrintManager::GetInstance().enn_clr_debug_zone(enn::debug::DbgPartition::kError),
              ENN_RET_SUCCESS);
    EXPECT_EQ(enn::debug::DbgPrintManager::GetInstance().enn_set_debug_zone(enn::debug::DbgPartition::kTest), ENN_RET_SUCCESS);
    EXPECT_EQ(enn::debug::DbgPrintManager::GetInstance().enn_clr_debug_zone(enn::debug::DbgPartition::kWarning),
              ENN_RET_SUCCESS);  // no effect
    EXPECT_EQ(enn::debug::DbgPrintManager::GetInstance().get_print_mask(), ZONE_BIT_MASK(enn::debug::DbgPartition::kTest));
    enn::debug::DbgPrintManager::GetInstance().set_mask(0xFFFFFFFF);
}

}  // namespace internal
}  // namespace test
}  // namespace enn
