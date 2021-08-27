#include <gtest/gtest.h>

#include <tuple>
#include <numeric>
#include <ctime>
#include <cstdlib>
#include <vector>

#include "runtime/client_process/client_process.hpp"


using namespace enn::runtime;
using namespace enn::identifier;

class ClientProcessTest : public testing::Test {
 protected:
    void SetUp() override {
    }
};


TEST_F(ClientProcessTest, create_client_process) {
    auto client_process = std::make_shared<ClientProcess>();
    EXPECT_EQ(client_process->get_id() >> 32, enn::util::get_caller_pid() & 0x1FFFF);
}

TEST_F(ClientProcessTest, create_and_release_10000_times) {
    constexpr int iter = 10000;
    for (int i = 0; i < iter; i++) {
        // ClientProcess is releasd on getting out of scope here.
        EXPECT_NO_THROW(std::make_shared<ClientProcess>());
    }
}

TEST_F(ClientProcessTest, throws_on_create_with_id_over) {
    constexpr int max = ClientProcess::UniqueID::Max;

    // add objects created to prevent from being released
    std::vector<ClientProcess::Ptr> em_list;

    // create ClientProcess objects by upper bound
    for (int i = 1; i <= max; i++) {
        em_list.push_back(std::make_shared<ClientProcess>());
    }
    // create one more ClientProcess object with id that exceeds the max limit.
    std::make_shared<ClientProcess>();
}
