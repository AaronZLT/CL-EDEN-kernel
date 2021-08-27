/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung
 * Electronics.
 */

#ifndef SRC_CLIENT_ENN_CLIENT_PARSER_H_
#define SRC_CLIENT_ENN_CLIENT_PARSER_H_

#include "model/parser/parser_utils.hpp"
#include "model/parser/cgo/cgo_utils.hpp"

static bool EnnClientCgoParse(std::unique_ptr<enn::EnnMemoryManager> &emm, enn::util::BufferReader::UPtrType &modelbuf,
                              const void *va, uint32_t size, uint32_t offset,
                              std::vector<std::shared_ptr<enn::EnnBufferCore>> *result) {
    auto parsed_params = enn::model::CgoUtils::parse_parameters(va, size);

    if (parsed_params == nullptr || parsed_params->size() == 0) {
        ENN_DBG_COUT << "Error: CGO parse_parameters" << std::endl;
        return false;
    }

    uint32_t fbs_section = offset + size;

    for (auto& param : *parsed_params) {
        std::shared_ptr<enn::EnnBufferCore> mem;


        if (param.size != 0 && param.is_loaded_from_file) {
            mem = emm->CreateMemory(param.size, enn::EnnMmType::kEnnMmTypeIon);
            if (!mem) {
                ENN_DBG_COUT << "Error: CreateMemory" << std::endl;
                return false;
            }
            ENN_DBG_COUT << "Load data from file for: " << param.name << std::endl;
            if (modelbuf->copy_buffer(reinterpret_cast<char *>(mem->va), param.size,
                                      param.offset + fbs_section + sizeof(enn::model::CGO::FileHeader::magic))) {
                ENN_DBG_COUT << "Error: Memory Load from file" << std::endl;
                return false;
            }
        } else {
            mem = emm->CreateMemoryObject(-1, 0, nullptr);  // generate null memory object
            if (!mem) {
                ENN_DBG_COUT << "Error: CreateMemoryObject" << std::endl;
                return false;
            }
        }

        result->push_back(mem);
    }

    return true;
}

#endif  // SRC_CLIENT_ENN_CLIENT_PARSER_H_