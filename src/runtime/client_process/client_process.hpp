#pragma once

#include <memory>
#include <string>

#include "common/identifier_decorator.hpp"
#include "common/identifier.hpp"
#include "common/enn_debug.h"
#include "runtime/pool/poolable.hpp"
#include "medium/enn_medium_utils.hpp"

namespace enn {
namespace runtime {

using namespace enn::identifier;

class ClientProcess : public pool::Poolable<IdentifierBase<FullIDType>> {
 private:
    using UniqueIDDeco = IdentifierDecorator<FullIDType, uint32_t, 32, 0x1FFFF>;

 public:
    using Ptr = std::shared_ptr<ClientProcess>;
    using ID = IdentifierBase<FullIDType>;
    class UniqueID : public UniqueIDDeco {
     public:
        // Bring members from base temlate class for client to access directly them.
        using UniqueIDDeco::Type;
        using UniqueIDDeco::Max;
        using UniqueIDDeco::Offset;
        using UniqueIDDeco::Mask;

        explicit UniqueID(const IdentifierBase<FullIDType>& base_id)
            : UniqueIDDeco(util::get_caller_pid() & Max, base_id) {}
    };

public:
    ClientProcess()
        : id_(std::make_unique<UniqueID>(reserved_id_)) {
            ENN_DBG_COUT << "A ClientProcess(ID: 0x" << *id_ << ") is created." << std::endl;
        }

    ~ClientProcess() {
        ENN_DBG_COUT << "A ClientProcess(ID: 0x" << *id_ << ") is released." << std::endl;
    }

    // @override a pure virtual function from the Poolable.
    bool operator==(const Poolable& rp) const override { return rp == *id_; }
    bool operator==(const ID& rid) const override { return *id_ == rid; }
    bool is_ancestor_of(const Poolable& rp) const override { return rp.is_descendant_of(*id_); }
    bool is_ancestor_of(const ID& rid) const override { return id_->is_overlapped_by(rid); }
    bool is_descendant_of(const Poolable& rp) const override { return rp.is_ancestor_of(*id_); }
    bool is_descendant_of(const ID& rid) const override { return id_->is_overlapping(rid); }
    std::string to_string() const override {
        std::stringstream s;
        s << "ClientProcess(ID: 0x" << id_->to_string() << ")";
        return s.str();
    }

    const ID& get_id() const {
        return *id_;
    }

 private:
    Identifier<FullIDType, 0x7FFF, 49> reserved_id_;
    ID::UPtr id_;  // client(caller)'s process id
};

};  // namespace enn
};  // namespace runtime