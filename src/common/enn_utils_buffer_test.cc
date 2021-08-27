#include "gtest/gtest.h"
#include "common/enn_utils_buffer.hpp"
#include "common/enn_debug.h"
#include "common/enn_utils.h"
#include <fstream>
#include <cstdio>

class ENN_GT_UTIL_BUFFER_TEST : public testing::Test {
  public:
    void SetUp() override {
        std::ofstream test_file(filename);
        test_file << testinput << std::endl;
        test_file.close();
    }

    void TearDown() override {
        std::remove(filename.c_str());
    }

    std::string testinput_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const char *testinput = testinput_str.c_str();
    std::string filename = "test.txt";
    std::ofstream test_file;
};

TEST_F(ENN_GT_UTIL_BUFFER_TEST, open_test_file_and_memory_load_and_check_size) {
    enn::util::BufferReader::UPtrType memoryobject = std::make_unique<enn::util::MemoryBufferReader>(testinput, testinput_str.size());
    enn::util::BufferReader::UPtrType fileobject = std::make_unique<enn::util::FileBufferReader>(filename);

    EXPECT_EQ(memoryobject->get_size(), testinput_str.size());  // doesn't contain newline char
    EXPECT_EQ(fileobject->get_size(), testinput_str.size() + 1);
}

TEST_F(ENN_GT_UTIL_BUFFER_TEST, open_test_load_to_buffer_full_size) {
    char out_test[100], out_test2[100];
    enn::util::BufferReader::UPtrType memoryobject = std::make_unique<enn::util::MemoryBufferReader>(testinput, testinput_str.size());
    enn::util::BufferReader::UPtrType fileobject = std::make_unique<enn::util::FileBufferReader>(filename);

    fileobject->copy_buffer(out_test);
    memoryobject->copy_buffer(out_test2);
}

TEST_F(ENN_GT_UTIL_BUFFER_TEST, open_test_load_to_buffer_partial) {
    char out_test[100];
    enn::util::BufferReader::UPtrType memoryobject = std::make_unique<enn::util::MemoryBufferReader>(testinput, 10);
    EXPECT_NE(memoryobject->copy_buffer(out_test, 5, testinput_str.size()), ENN_RET_SUCCESS);
    EXPECT_NE(memoryobject->copy_buffer(out_test, testinput_str.size(), 1), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_UTIL_BUFFER_TEST, open_test_load_to_buffer_partial_compare) {
    char out_test[100], out_test2[100];
    enn::util::BufferReader::UPtrType memoryobject = std::make_unique<enn::util::MemoryBufferReader>(testinput, testinput_str.size());
    enn::util::BufferReader::UPtrType fileobject = std::make_unique<enn::util::FileBufferReader>(filename);

    memoryobject->copy_buffer(out_test, 10, 10);
    fileobject->copy_buffer(out_test2, 10, 10);

    EXPECT_EQ(memcmp(out_test, out_test2, 10), 0);
}

static void testCursor(std::unique_ptr<enn::util::BufferReader>& obj) {
    char out_test[100];

    EXPECT_NE(obj->get_size(), 0);
    EXPECT_EQ(obj->get_cursor(), 0);
    // over sized = error
    EXPECT_NE(obj->set_cursor(10990), ENN_RET_SUCCESS);
    EXPECT_EQ(obj->set_cursor(2), ENN_RET_SUCCESS);
    EXPECT_EQ(obj->get_cursor(), 2);
    obj->copy_buffer(out_test, 2);
    out_test[2] = 0;
    EXPECT_STREQ(out_test, "AB");
    obj->copy_buffer_with_cursor(out_test, 6);
    out_test[6] = 0;
    EXPECT_STREQ(out_test, "CDEFGH");
    obj->copy_buffer_with_cursor(out_test, 14);
    out_test[14] = 0;
    EXPECT_STREQ(out_test, "IJKLMNOPQRSTUV");
    obj->copy_buffer_with_cursor(out_test, 44);
    EXPECT_STREQ(out_test, "IJKLMNOPQRSTUV");  // no changed from error
}

TEST_F(ENN_GT_UTIL_BUFFER_TEST, test_cursor_set_get) {
    enn::util::BufferReader::UPtrType memoryobject = std::make_unique<enn::util::MemoryBufferReader>(testinput, testinput_str.size());
    enn::util::BufferReader::UPtrType fileObject = std::make_unique<enn::util::FileBufferReader>(filename);
    testCursor(memoryobject);
    testCursor(fileObject);
}
