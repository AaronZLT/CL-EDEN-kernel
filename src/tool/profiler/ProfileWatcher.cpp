/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file    ProfileWatcher.cpp
 * @brief   It defines class of ProfileWatcher.
 * @details A thread created creates profile-tree and prints the result in the way of DFS.
 * @version 1
 */

#include <fstream>

#include "tool/profiler/include/ProfileWatcher.hpp"

class ProfileWatcher;

ProfileWatcher::ProfileWatcher(uint64_t id)
: id(id) {
    profile_thread_should_finish() = false;
    profile_data_is_normal() = true;
    create_profile_tree();
    create_profile_data_q();
    watcher_thread = std::thread(&ProfileWatcher::thread_func, this);
}

ProfileWatcher::~ProfileWatcher() {
    profile_thread_should_finish() = true;
    cv_.notify_all();
    watcher_thread.join();
    if (profile_data_is_normal()) {
        try {
            dump_profile_tree();
            trim_profile_tree();
            print_profile_tree();
        } catch (const std::exception& e) {
            profile::log::error("%s\n", e.what());
        }
    }
    else {
        print_profile_data_error();
    }
    delete profile_data_q;
    delete profile_tree_root;
}

std::condition_variable& ProfileWatcher::get_condition_variable() {
    return cv_;
}

void ProfileWatcher::push_profile_data(ProfileData* profile_data) {
    if (!profile_data_q) {
        profile::log::error("Queue for profiled data is null.\n");
        profile::log::error("Make sure that you did not call the PROFILE_* macro before calling START_PROFILER.\n");
        return;
    }
    try {
        profile_data_q->push(profile_data);
    } catch (const std::exception& e) {
        profile::log::error("Profile data push failed : %s \n", e.what());
    }
    return;
}

void ProfileWatcher::print_profile_tree() {
    /*
     * Output stream is decided, according to The value from vendor.enn.profile.out.
     */
    profile::log::result(MSG_TABLE_BORDER_TOP, id);
    profile::log::result(MSG_TABLE_COLUMN_TITLE);
    for (auto sub_tree : profile_tree_root->get_children()) {
        // Recursive call for DFS
        print_node(sub_tree, sub_tree, 1);
    }

    profile::log::result(MSG_TABLE_BORDER_BOTTOM);
    fflush(stdout);
    return;
}

void ProfileWatcher::print_node(ProfileTreeNode* sub_tree_root, ProfileTreeNode* node, int level) {
    if (level > 0) {
        // TODO(sahil.sharma): Control printing excluded node by checking config
        //  ,which is "EXYNOS_NN_PROFILER_OUTPUT_EXCLUDED".
        if (node->is_excluded) return;
        std::string sep;
        // Below seperator is for seperating the releation between parent and child.
        for (int i = 1; i <= level; i++) {
            sep += "_";
        }
        // Calculate the proportion deviding "self duration" by "total duration"
        node->calc_proportion(sub_tree_root);
        // print the profiled data.
        profile::log::result(MSG_CONTENT_ROW,
        node->get_avg_duration().count(), node->get_min_duration().count(),
        node->get_max_duration().count(), node->get_self_duration().count(),
        node->get_proportion_of_total(),  node->get_ninetieth_latency(), node->get_iteration(),
        sep.c_str(), node->get_label().c_str());
    }
    for (auto& child : node->get_children()) {
        print_node(sub_tree_root, child, level + 1);
    }
    delete node;
    return;
}

void ProfileWatcher::print_profile_data_error() {
    profile::log::error("An error occurred while processing the profile data.\n");
    profile::log::error("Look into the log above, and you can uncover the cause.\n");
    return;
}

void ProfileWatcher::calculate_self_duration(ProfileTreeNode* node) {
    std::chrono::microseconds sum_of_duration = std::chrono::microseconds::zero();
    for (auto child : node->get_children()) {
        calculate_self_duration(child);
        // The duration to be excluded is already subtracted.
        if (child->is_excluded) continue;
        sum_of_duration += child->get_avg_duration();
    }
    node->calc_avg_duration();
    node->set_self_duration(node->get_avg_duration() - sum_of_duration);
    node->calc_ninetieth_latency();
    return;
}

void ProfileWatcher::trim_profile_tree() {
    for (auto sub_tree : profile_tree_root->get_children()) {
        calculate_self_duration(sub_tree);
    }
    return;
}

void ProfileWatcher::thread_func() {
    while (!profile_thread_should_finish()|| !profile_data_q->is_empty()) {
        std::unique_lock<std::mutex> lk(m_);
        // The latency for calling notify is included in execution time
        // That's why wait_for is used for time out.
        // Notify is callsed only when destructing the profile_watcher.
        cv_.wait_for(lk, std::chrono::microseconds(WATCHER_THREAD_WAIT_TIME));
        if (!supervise_profile_tree()) {
            profile_data_is_normal() = false;
            return;
        }
    }
    return;
}

bool& ProfileWatcher::profile_thread_should_finish() {
    static bool finish;
    return finish;
}

bool& ProfileWatcher::profile_data_is_normal() {
    static bool is_normal;
    return is_normal;
}

void ProfileWatcher::create_profile_tree() {
    profile_tree_root = new ProfileTreeNode("DUMMY_ROOT");
    profile_tree_head = profile_tree_root;
    return;
}

void ProfileWatcher::create_profile_data_q() {
    profile_data_q = new ProfileDataQueue<ProfileData>();
    return;
}

std::string ProfileWatcher::parse_label(ProfileData* profile_data) {
    std::string label;
    if (profile_data->get_custom_label().empty()) {
        std::size_t name_found = profile_data->get_file().find_last_of("/");
        std::string pure_file_name = profile_data->get_file().substr(name_found + 1);
        label = profile_data->get_func() + "(" + pure_file_name + ":" + profile_data->get_line() + ")";
    }
    else if (profile_data->get_op_num() > 0) {
        label = profile_data->get_custom_label() + "_" + std::to_string(profile_data->get_op_num());
    } else {
        label = profile_data->get_custom_label();
    }
    return label;
}

void ProfileWatcher::update_tree_foundation(const std::string& label, ProfileTreeNode* node) {
    map_info_oftree_foundation.insert(std::make_pair(label, node));
    return;
}

ProfileTreeNode* ProfileWatcher::search_tree_foundation(const std::string& label) {
    auto map_iter = map_info_oftree_foundation.find(label);
    if (map_iter == map_info_oftree_foundation.end()) return nullptr;
    else return map_iter->second;
}

// Propagate the duration to be needed to subtarct from duration of parent.
void ProfileWatcher::propagate_excluded_duration(ProfileTreeNode* node) {
    auto duration = node->get_cur_duration();
    auto parent_node = node->get_parent();
    while (parent_node) {
        parent_node->update_exclude_duration(duration);
        parent_node = parent_node->get_parent();
    }
    return;
}

bool ProfileWatcher::process_entry_profile_data(ProfileData* profile_data) {
    if (profile_data == nullptr) {
        profile::log::error("The profile_data is nullptr.\n");
        return false;
    }
    EntryProfileData* entry_profile_data = dynamic_cast<EntryProfileData*>(profile_data);
    if (entry_profile_data == nullptr) {
        profile::log::error("The entry_profile_data is nullptr.\n");
        return false;
    }
    std::string label = parse_label(entry_profile_data);
    auto node_found = search_tree_foundation(label);
    // node_found is nullptr : this scope to be tried to be profiled is initialized to profile tree.
    if (node_found == nullptr) {
        profile_tree_head = profile_tree_head->push_child_node(
            new ProfileTreeNode(label,
                                entry_profile_data->get_time(),
                                entry_profile_data->get_thread_id(),
                                entry_profile_data->is_excluded));
        update_tree_foundation(label, profile_tree_head);
    // node_found is not nullptr : add only entry time point in this profiled data to existing node.
    } else {
        node_found->add_entry_time(entry_profile_data->get_time());
    }
    return true;
}

bool ProfileWatcher::process_exit_profile_data(ProfileData* profile_data) {
    if (profile_data == nullptr) {
        profile::log::error("The profile_data is nullptr.\n");
        return false;
    }
    ExitProfileData* exit_profile_data = dynamic_cast<ExitProfileData*>(profile_data);
    std::string label = parse_label(exit_profile_data);
    auto node_found = search_tree_foundation(label);
    // node_found is nullptr: means there is no entry node added before, which is error.
    if (node_found == nullptr) {
        profile::log::error("The end point of the scope to profile(%s) is not matched.\n", label.c_str());
        profile::log::error("Please check if profiling the entry data frrom same scope is called.\n");
        return false;
    // node_found is not nullptr : normal exit data is added, which matches to entry data.
    } else {
        // This exit data is from the scope of first iteration, so it should be added to current tree head.
        if (node_found->is_established == false) {
            if (profile_tree_head->get_label() == label) {
                profile_tree_head = profile_tree_head->set_duration(exit_profile_data->get_time())->pop_child();
                node_found->is_established = true;
            // Current profile_tree_head is not matched this exit profiled data.
            // That means these scopes in this sub-tree are not executed synchronously.
            } else {
                profile::log::error("The end point of the scope to profile(%s) is not matched.\n", label.c_str());
                profile::log::error("Please check if these scope is in the synchronous flow.\n");
                return false;
            }
        // This exit data is from interation of 2 or more times.
        } else {
            node_found->set_duration(exit_profile_data->get_time());
        }
        // Propagate the duration to be excluded to parent nodes if there is.
        if (node_found->is_excluded) {
            propagate_excluded_duration(node_found);
        }
    }
    return true;
}

bool ProfileWatcher::process_calculated_profile_data(ProfileData* profile_data) {
    if (profile_data == nullptr) {
        profile::log::error("The profile_data is nullptr.\n");
        return false;
    }

    CalculatedProfileData* calculated_profile_data = dynamic_cast<CalculatedProfileData*>(profile_data);
    if (calculated_profile_data == nullptr) {
        profile::log::error("The calculated_profile_data is nullptr.\n");
        return false;
    }
    CalculatedProfileNode* calculated_profile_node = calculated_profile_data->get_calculated_profile_node();
    // The label of profile_data is the label of root in calculated sub tree.
    calculated_profile_data->set_custom_label(calculated_profile_node->label);
    std::string label = parse_label(calculated_profile_data);
    auto node_found = search_tree_foundation(label);
    // The node_found is nullptr means this sub-tree of calculated_profile_data should be initialized in the profile_tree
    // The node_found is not nullptr means this sub-tree of calculated_profile_data should be updated to the profile_tree.
    if (node_found == nullptr) {
        // It dosen't need to update position pointed by profile_tree_head.
        // It have only to append the tree as child to current profile_tree_head.
        ProfileTreeNode* temp_profile_tree_head = profile_tree_head;
        auto node = append_calculated_profile_node(calculated_profile_node, temp_profile_tree_head);
        update_tree_foundation(label, node);
    } else {
        update_calculated_profile_node(calculated_profile_node, node_found);
    }
    return true;
}

ProfileTreeNode* ProfileWatcher::append_calculated_profile_node(CalculatedProfileNode* calculated_profile_node, ProfileTreeNode* head) {
    // Parse the calculated_proifle_node as sub-tree and append all of nodes to the ProfileTree.
    ProfileTreeNode* root_of_subtree_appended = head;
    head = head->push_child_node(
            new ProfileTreeNode(
                calculated_profile_node->label, calculated_profile_node->duration));
    // Delete the memory of CalculatedProfileNode after copying to ProfileTreeNode.
    // The free function is used for handling the dynamic allocation from both c and c++.
    free(calculated_profile_node->label);
    calculated_profile_node->label = nullptr;
    if (profile_tree_head == root_of_subtree_appended) root_of_subtree_appended = head;
    if (calculated_profile_node->child != nullptr) {
        for (uint8_t i = 0; calculated_profile_node->child[i] != nullptr; i++) {
            append_calculated_profile_node(calculated_profile_node->child[i], head);
        }
        free(calculated_profile_node->child);
        calculated_profile_node->child = nullptr;
    }
    free(calculated_profile_node);
    calculated_profile_node = nullptr;
    return root_of_subtree_appended;
}

void ProfileWatcher::update_calculated_profile_node(CalculatedProfileNode* calculated_profile_node, ProfileTreeNode* head) {
    // Parse the calculated_proifle_node as sub-tree and apply them to existing sub tree.
    *head = std::chrono::microseconds(calculated_profile_node->duration);
    // Delete the memory of CalculatedProfileNode after updating the duration to the ProfileTreeNode.
    // The free function is used for handling the dynamic allocation from both c and c++.
    free(calculated_profile_node->label);
    calculated_profile_node->label = nullptr;
    if (calculated_profile_node->child != nullptr) {
        for (uint8_t i = 0; calculated_profile_node->child[i] != nullptr; i++) {
            update_calculated_profile_node(calculated_profile_node->child[i], head->get_children().at(i));
        }
        free(calculated_profile_node->child);
        calculated_profile_node->child = nullptr;
    }
    delete calculated_profile_node;
    calculated_profile_node = nullptr;
    return;
}

bool ProfileWatcher::supervise_profile_tree() {
    while (!profile_data_q->is_empty()) {
        ProfileData* profile_data = profile_data_q->pop();
        if (profile_data == nullptr) {
            return true;
        }
        bool ret = true;
        if (typeid(*profile_data) == typeid(EntryProfileData)) {
            ret = process_entry_profile_data(profile_data);
        } else if (typeid(*profile_data) == typeid(ExitProfileData)) {
            ret = process_exit_profile_data(profile_data);
        } else if (typeid(*profile_data) == typeid(CalculatedProfileData)) {
            ret = process_calculated_profile_data(profile_data);
        }
        delete profile_data;
        if (!ret) return false;  // abnormal return
    }
    return true;
}

void ProfileWatcher :: dump_node(ProfileTreeNode* node, std::string label) {
    std::ofstream out_dump;
    out_dump.open(DUMP_PATH, std::ofstream::app || std::ofstream::out);
    std::vector<std::chrono::microseconds> durations = node->get_durations();
    out_dump<<(label + node->get_label());
    for(int32_t idx = 0; idx < durations.size(); idx++) {
        out_dump<<","<<std::to_string(durations[idx].count());
    }
    out_dump<<"\n";
    out_dump.close();
    for(auto& child : node->get_children()) {
        dump_node(child, label+"_");
    }
}

void ProfileWatcher :: dump_profile_tree() {
    char model_info[256];
    std::sprintf(model_info, MSG_TABLE_BORDER_TOP, id);
    std::ofstream out_dump;
    out_dump.open(DUMP_PATH, std::ofstream::app || std::ofstream::out);
    out_dump<<model_info<<"\n";
    out_dump.close();
    // Use DFS to recursively dump the profile subtrees
    for (auto sub_tree : profile_tree_root->get_children()) {
        dump_node(sub_tree, "_");
    }
}
