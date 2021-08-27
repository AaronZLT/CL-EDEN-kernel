#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <functional>


#include "model/memory/buffer_table.hpp"

using namespace enn::model::memory;

class BufferTest : public testing::Test {
 protected:
    void SetUp() override {
        create_buffer_table();
    }

    void create_buffer_table() {
        // create int pointer variables that are dynamically allocated.
        std::vector<int*> int_ptr_vec;
        // value of int object is 0~9.
        // TODO(yc18.cho): implement to take variables as argument.
        for (int i = 0; i < 10; i++) {
            int_ptr_vec.push_back(new int{i});
        }
        int index = 0;
        size_t size = 0;  // suppose that size is same as index.
        // Index and size of int object is also 0~9
        // i.e. value == index == size
        for (auto int_ptr : int_ptr_vec) {
            buffer_table.add(index++, int_ptr, size++);
        }
    }

    void create_indexed_buffers() {
        size_t size = 0;
        int32_t index = 0;
        for ( ; index < 10; index++, size++) {
            indexed_buffers.push_back({index, size});
        }
    }

    BufferTable buffer_table;
    std::vector<IndexedBuffer> indexed_buffers;
};


TEST_F(BufferTest, check_buffer_table_with_index) {
    for (int i = 0; i < 10; i++) {
        auto buffer = buffer_table[i];
        auto int_val = *(static_cast<const int*>(buffer.get_addr()));
        EXPECT_EQ(int_val, i);
        delete buffer.get_addr();
    }
}

TEST_F(BufferTest, check_buffer_table_with_indexed_buffer) {
    create_indexed_buffers();
    for (int i = 0; i < 10; i++) {
        auto buffer = buffer_table[indexed_buffers[i]];
        auto int_val = *(static_cast<const int*>(buffer.get_addr()));
        EXPECT_EQ(int_val, i);
        delete buffer.get_addr();
    }
}
