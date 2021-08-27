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

#ifndef SRC_COMMON_ENN_LOG_HPP_
#define SRC_COMMON_ENN_LOG_HPP_

#include <sstream>
#include <string>
#include <iostream>
#include <memory>

namespace enn {
namespace debug {

int __attribute__((format(printf, 6, 7)))
enn_print_with_check(const char *log_tag, DbgPartition zone, DbgPrintOption check, const char *caller, const int caller_num,
                         const char *format, ...);
class EnnMsgHandler {
public:
    EnnMsgHandler(const char* func, const int line, DbgPartition type)
        : _type(type), _func(func), _line(line) {
        _msg = std::make_shared<std::stringstream>();
    }

    template<typename T>
    EnnMsgHandler&  operator << (T&& arg) {
        (*_msg) << std::forward<T>(arg);
        return *this;
    }

    // std::ostream& (*)(std::ostream&)) is equivalent to std::endl
    void operator << (std::ostream& (*)(std::ostream&)) {
        enn_print_with_check(nullptr, _type, DbgPrintOption::kEnnPrintTrue, _func, _line, "  %s\n", c_str());
    }

    const char *c_str() const {
        return str().c_str();
    }

    std::string & str() const {
        my_str = (*_msg).str();
        return my_str;
    }

 private:
    std::shared_ptr<std::stringstream> _msg;
    DbgPartition _type;
    mutable std::string my_str;
    const char* _func;
    const int _line;
};

}  //  namespace debug
}  //  namespace enn

#endif  // SRC_COMMON_ENN_LOG_HPP_
