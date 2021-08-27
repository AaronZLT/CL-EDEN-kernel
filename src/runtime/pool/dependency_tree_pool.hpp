#pragma once

#include <memory>
#include <mutex>
#include <functional>
#include <list>
#include <algorithm>
#include <array>
#include <string>

#include "runtime/pool/poolable.hpp"
#include "common/extended_type_traits.hpp"
#include "common/enn_debug.h"

namespace enn {
namespace runtime {
namespace pool {

// @Description
// - There are mandatory prerequisites for objects that can be kept in a pool
//    a) The template argument, the SP should be std::shared_ptr type.
//    b) The template argument, the SP should be class that inherits from Poolable interface
//        and overrides pure virtual functions in it.
// - It is an object pool in the form of a dependency tree that guarantees thread safety.
// - It is a one-way tree that composes with the inner class "Node" which points to children nodes.
// - It is a tree in which a dependency between nodes must always be maintained.
//    a) When a node is added, the node it depends on must be in the tree.
//    b) When trying to find a node, it traverses through the dependency path from the root of the tree.
//    c) When a node is removed, all child nodes of that node are also deleted.
// - It creates and finds dependency paths by calling virtual functions in the Poolable interface.
// - It follows the general terminology of a Tree ADT.
template <typename SP,
          size_t Height,
          typename E = typename SP::element_type,
          typename = typename std::enable_if_t<enn::util::is_shared_ptr<SP>::value>,
          typename = typename std::enable_if_t<std::is_base_of<Poolable<typename E::Key>, E>::value>>
class DependencyTreePool {
 private:
    class Node {
     public:
        using Ptr = std::shared_ptr<Node>;
        using WPtr = std::weak_ptr<Node>;
        // Array to keep the width each on a certain level in a Tree.
        using WIDTH_LIST = std::array<size_t, Height + 1>;

     private:
        inline static WIDTH_LIST width_list_;

     public:
        static WIDTH_LIST& get_width_list() { return width_list_; }

     public:
        Node() : object_{nullptr}, level_{0} { width_list_[level_]++; }
        Node(const SP& object, size_t level) : object_{object}, level_{level} { width_list_[level_]++; }
        ~Node() { width_list_[level_]--; }

        bool operator==(const Node& rhs) { return object_ == *rhs; }
        const SP& get() const { return object_; }
        const std::list<Ptr>& get_children() { return children_; }
        size_t get_level() { return level_; }
        template <typename C> void add_child(C&& child) { children_.push_back(std::forward<C>(child)); }
        void remove_child(const Ptr& node) { children_.remove_if([&](const Ptr& child) { return child == node; }); }
        std::string to_string() const {
            if (!object_) {
                return "root-node";
            }
            return object_->to_string() + "(level:" + std::to_string(level_) + ")";
        }

     private:
        SP object_;
        size_t level_;
        std::list<Ptr> children_;
    };

 private:
    typename Node::Ptr root_;
    mutable std::mutex mutex_;

 private:
    inline typename Node::Ptr find_impl(const typename Node::Ptr& parent,
                                        const typename E::Key& key,
                                        size_t target_level,
                                        size_t current_level) const {
        for (auto& child : parent->get_children()) {
            if (target_level == current_level && *child->get() == key) {
                ENN_DBG_COUT << "An Object(Ox" << key << ") is found in children of "
                << parent->to_string() << std::endl;
                return child;
            } else if (child->get()->is_ancestor_of(key)) {
                return find_impl(child, key, target_level, current_level + 1);
            }
        }
        ENN_ERR_COUT << "An Object with the ID(0x" << key
                     << ") is not found in the children of "<< parent->to_string() << std::endl;
        return nullptr;
    }

    inline bool add_impl(const typename Node::Ptr& parent,
                               const SP& object,
                               size_t target_level,
                               size_t current_level) const {
        if (target_level == current_level) return append_node(parent, object);
        for (auto& child : parent->get_children()) {
            if (child->get()->is_ancestor_of(*object))
                return add_impl(child, object, target_level, current_level + 1);
        }
        ENN_ERR_COUT << "An Object on which " << object->to_string()
                     << " to be added depends is not found in children of " << parent->to_string() << std::endl;
        return false;
    }

    inline bool append_node(const typename Node::Ptr& parent, const SP& object) const {
        auto& children = parent->get_children();
        auto it = std::find_if(children.begin(), children.end(),
                               [&](const typename Node::Ptr& child) { return child->get() == object; });
        // Return false if the object to add already exists.
        if (it != children.end()) {
            ENN_ERR_COUT << object->to_string() << " already exists as a child of "
                         << parent->to_string() << std::endl;
            return false;
        }
        parent->add_child(std::make_shared<Node>(object, parent->get_level() + 1));
        ENN_DBG_COUT << object->to_string() <<" is added to " << parent->to_string() << std::endl;
        return true;
    }

    inline bool remove_impl(const typename Node::Ptr& parent,
                            const typename E::Key& key,
                            size_t target_level,
                            size_t current_level) const {
        for (auto& child : parent->get_children()) {
            ENN_INFO_COUT << "child is " << child->to_string() << std::endl;
            if (target_level == current_level && *child->get() == key) {
                parent->remove_child(child);
                ENN_DBG_COUT << "An Object(Ox" << key << ") is removed from children of "
                << parent->to_string() << std::endl;
                return true;
            } else if (child->get()->is_ancestor_of(key)) {
                return remove_impl(child, key, target_level, current_level + 1);
            }
        }
        ENN_ERR_COUT << "An Object on which an Object(ID:0x" << key
                     << ") to be removed depends is not found in children of " << parent->to_string() << std::endl;
        return false;
    }

 public:
    DependencyTreePool() : root_{std::make_shared<Node>()} {}

    template <size_t level>
    bool add(const SP& object) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!add_impl(root_, object, level, 1)) {
            return false;
        }
        return true;
    }

    template <size_t level>
    SP find(const typename E::Key& key) const {
        std::lock_guard<std::mutex> guard(mutex_);
        auto found = find_impl(root_, key, level, 1);
        if (!found) return nullptr;
        return found->get();
    }

    template <size_t level>
    bool remove(const typename E::Key& key) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!remove_impl(root_, key, level, 1)) return false;
        return true;
    }

    template <size_t level>
    size_t width() const {
        std::lock_guard<std::mutex> guard(mutex_);
        return Node::get_width_list()[level];
    }
};

};  // namespace pool
};  // namespace runtime
};  // namespace enn
