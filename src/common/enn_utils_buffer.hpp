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

#ifndef SRC_COMMON_ENN_UTILS_BUFFER_HPP_
#define SRC_COMMON_ENN_UTILS_BUFFER_HPP_

#include "common/enn_utils.h"
#include <cstdio>
#include <memory>
#include <cstring>

namespace enn {
namespace util {

class BufferReader {
  public:
    using UPtrType = std::unique_ptr<BufferReader>;
    uint32_t get_size() {
        return _size;
    }

    virtual bool is_valid() = 0;

    virtual EnnReturn copy_buffer(char *out_addr, uint32_t size = 0, uint32_t offset = 0) {
        CHECK_AND_RETURN_ERR(size + offset > _size, ENN_RET_INVAL, "size(%d) + offset(%d) is bigger than original(%d)\n",
                             size, offset, _size);
        CHECK_AND_RETURN_ERR(out_addr == nullptr, ENN_RET_INVAL, "output addr is null\n");
        CHECK_AND_RETURN_ERR((size == 0 && offset != 0), ENN_RET_INVAL, "Offset(%d) should be zero if size = 0\n", offset);

        return ENN_RET_SUCCESS;
    }

    virtual EnnReturn copy_buffer_with_cursor(char *out_addr, uint32_t move) {
        CHECK_AND_RETURN_ERR(move + _current_cursor > _size, ENN_RET_INVAL, "move(%d), current(%d) > size(%d)\n", move,
                             _current_cursor, _size);
        if (copy_buffer(out_addr, move, _current_cursor)) return ENN_RET_INVAL;
        _current_cursor += move;

        return ENN_RET_SUCCESS;
    }

    virtual EnnReturn set_cursor(uint32_t point) {
        CHECK_AND_RETURN_ERR(point > _size, ENN_RET_INVAL, "parameter(%d) is too big\n", point);
        _current_cursor = point;
        return ENN_RET_SUCCESS;
    }

    virtual uint32_t get_cursor(void) {
        return _current_cursor;
    }

    virtual EnnReturn move_cursor(uint32_t offset) {
        CHECK_AND_RETURN_ERR(_current_cursor + offset > _size, ENN_RET_INVAL, "parameter(%d) is too big\n", offset);
        _current_cursor += offset;
        return ENN_RET_SUCCESS;
    }

    virtual ~BufferReader() {}

  protected:
    uint32_t _size = 0;
    uint32_t _current_cursor = 0;
};

class FileBufferReader : public BufferReader {
  public:
    FileBufferReader() = delete;
    explicit FileBufferReader(std::string filename) : _filename(filename) {
        fd = fopen(filename.c_str(), "rb");
        _size = get_file_size();
        ENN_INFO_COUT << "File " << filename << " loaded. (fd: " << fd << ", size: " << _size << ")" << std::endl;
    }

    bool is_valid() { return fd != nullptr; }

    ~FileBufferReader() {
        if (fd)
            fclose(fd);
    }

    EnnReturn copy_buffer(char *out_addr, uint32_t size = 0, uint32_t offset = 0) override {
        if (BufferReader::copy_buffer(out_addr, size, offset))  return ENN_RET_INVAL;
        if (size == 0)   size = BufferReader::_size;

        CHECK_AND_RETURN_ERR((fseek(fd, offset, SEEK_SET) != 0), ENN_RET_IO, "File Open Error\n");
        CHECK_AND_RETURN_ERR((size != static_cast<int>(fread(out_addr, sizeof(char), size, fd))), ENN_RET_IO,
                             "File Open Error\n");

        return ENN_RET_SUCCESS;
    }

  private:
    std::string _filename;
    std::FILE *fd = nullptr;

    uint32_t get_file_size() {
        CHECK_AND_RETURN_ERR(fd == nullptr, 0, "File open error\n");
        if (_size != 0) return _size;
        int size = 0;

        fseek(fd, 0, SEEK_END);
        size = ftell(fd);

        CHECK_AND_RETURN_ERR(size <= 0, 0, "File size is not correct(%d)\n", size);
        return size;
    }
};

class MemoryBufferReader : public BufferReader {
  public:
    MemoryBufferReader() = delete;
    explicit MemoryBufferReader(const char *base_addr, const uint32_t size) : _base_addr(base_addr) {
        BufferReader::_size = size;
        ENN_INFO_COUT << "Buffer " << std::hex << base_addr << std::dec << "(" << size << ") loaded." << std::endl;
    }

    bool is_valid() {
        return _base_addr != nullptr;
    }

    EnnReturn copy_buffer(char *out_addr, uint32_t size = 0, uint32_t offset = 0) override {
        if (BufferReader::copy_buffer(out_addr, size, offset))
            return ENN_RET_INVAL;
        if (size == 0)
            size = BufferReader::_size;

        memcpy(out_addr, _base_addr + offset, size);
        return ENN_RET_SUCCESS;
    }

  private:
    const char *_base_addr;
};
}  // namespace util
}  // namespace enn

#endif  // SRC_COMMON_ENN_UTILS_BUFFER_HPP_
