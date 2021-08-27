#pragma once

#include <mutex>
#include <functional>
#include <bitset>
#include <limits>

#include "common/identifier_base.hpp"
#include "common/enn_utils.h"

namespace enn {
namespace identifier {


// Note:
// For the sake of code reuse, this IdentifierDecorator takes data of
//  concrete decorator class via template parameters.
//  virtual methods are performed in the manner of the general decorator pattern,
//  using data injected by derived concrete class.
//  If the implementation of the interface function does not change when adding a derived class,
//  simply add data values needed for derived class ​​as template parameters.
//  Otherwise, you only need to override virtual interface functions in your derived class.
template <typename BaseIDType, typename UniqueIDType,
          size_t UniqueIDOffset, UniqueIDType UniqueIDMax = std::numeric_limits<UniqueIDType>::max()>
class IdentifierDecorator : public IdentifierBase<BaseIDType> {
 public:
    using Type = UniqueIDType;
    static constexpr Type Max = UniqueIDMax;
    static constexpr size_t Offset = UniqueIDOffset;
    static constexpr BaseIDType Mask = static_cast<BaseIDType>(Max) << Offset;

    explicit IdentifierDecorator(const IdentifierBase<BaseIDType>& base_id)
        : unique_id_{0}, mask_on_{0}, mask_off_{0}, base_id_{base_id}, is_acquired_from_pool_{true} {
            // acquire the unique_id_ from reusable_id_pool bitset.
            acquire_reusable_id();
            load_eagerly();
    }

    // It doesn't acquire unique_id_ from reusable_id_pool.
    //  Instead, it sets unique_id passed as parameter to unique_id_.
    IdentifierDecorator(Type unique_id, const IdentifierBase<BaseIDType>& base_id)
        : unique_id_{unique_id}, mask_on_{0}, mask_off_{0}, base_id_{base_id}, is_acquired_from_pool_{false} {
            load_eagerly();
    }

    virtual ~IdentifierDecorator() {
        // release the unique_id to reusable_id_pool only if it was acquired from it.
        if (is_acquired_from_pool_) release_reusable_id();
    }

    BaseIDType get() const override {
        if (!this->id_) {
            this->id_ = base_id_.get() | (static_cast<BaseIDType>(unique_id_) << Offset);
        }
        return this->id_;
    }

    operator BaseIDType() const override {
        return get();
    }

    BaseIDType mask_on() const override {
        if (!mask_on_) {
            mask_on_ = base_id_.mask_on() | Mask;
        }
        return mask_on_;
    }

    BaseIDType mask_off() const override {
        if (!mask_off_) {
            mask_off_ = base_id_.mask_off() & ~Mask;
        }
        return mask_off_;
    }

 private:
    // Helper function for eagar loading of members taken by virtual function call on base_id_.
    void load_eagerly() {
        get();
        mask_on();
        mask_off();
    }

    // reusable_id_pool(std::bitset) is a kind of reusable ID pool.
    //  It is local static variable that all objcets can share it in order to acquire or release reusable id from it.
    static std::bitset<Max>& get_reusable_id_pool() {
        static std::bitset<Max> reusable_id_pool;
        return reusable_id_pool;
    }

    // return static mutex for reusable_id_pool
    static std::mutex& get_mutex_for_id_pool() {
        static std::mutex mutex;
        return mutex;
    }

    // acquire an available id from reusable_id_pool(bitset).
    void acquire_reusable_id() {
        std::lock_guard<std::mutex> guard(get_mutex_for_id_pool());
        if (get_reusable_id_pool().all()) {
            throw std::runtime_error("Error: Exceeds the maximum number of ID that can be created.");
        }
        static std::size_t idx = -1;  // most recently used index
        while (true) {
            ++idx %= get_reusable_id_pool().size();  // circular count-up
            if (!get_reusable_id_pool().test(idx)) {
                unique_id_ = idx + 1;
                get_reusable_id_pool().set(idx);
                break;
            }
        }
    }

    // release an used id to reusable_id_pool(bitset).
    void release_reusable_id() {
        std::lock_guard<std::mutex> guard(get_mutex_for_id_pool());
        try {
            get_reusable_id_pool().reset(unique_id_ - 1);
        } catch (const std::exception& e) {
            ENN_ERR_COUT << e.what() << std::endl;
        }
    }

 private:
    Type unique_id_;
    mutable BaseIDType mask_on_;
    mutable BaseIDType mask_off_;
    const IdentifierBase<BaseIDType>& base_id_;
    bool is_acquired_from_pool_;  // flag if the unique_id_ is acquired from reusable_id_pool bitset.
};


};  // namespace identifier
};  // namespace enn
