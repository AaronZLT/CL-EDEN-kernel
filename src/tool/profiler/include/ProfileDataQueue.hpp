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
 * @file    ProfileDataQueue.hpp
 * @brief   It declares the classes to enqueue or dequeue the instance from ProfileData.
 * @details ProfileWatcher object holds a ProfileDataQueue as class variable.
 * @version 1
 */

#ifndef TOOLS_PROFILER_INCLUDE_PROFILEDATAQUEUE_HPP_
#define TOOLS_PROFILER_INCLUDE_PROFILEDATAQUEUE_HPP_

#ifdef __cplusplus

#include <condition_variable>
#include <queue>
#include <mutex>

template <typename T>
class ProfileDataQueue
{
 public:
    ProfileDataQueue() = default;
    ProfileDataQueue(const ProfileDataQueue&) = delete;
    ProfileDataQueue& operator=(const ProfileDataQueue&) = delete;

    T* pop();
    void pop(T* item);
    void push(T* item);
    bool is_empty();


 private:
    std::queue<T*> queue_;
    std::mutex mutex_;
};

template <typename T>
T* ProfileDataQueue<T>::pop() {
    std::unique_lock<std::mutex> mlock(mutex_);
    if (queue_.empty()) return nullptr;
    T* item = queue_.front();
    queue_.pop();
    return item;
}

template <typename T>
void ProfileDataQueue<T>::pop(T* item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    if (queue_.empty()) return;
    item = queue_.front();
    queue_.pop();
    return;
}

template <typename T>
void ProfileDataQueue<T>::push(T* item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(item);
    return;
}

template <typename T>
bool ProfileDataQueue<T>::is_empty() {
    std::unique_lock<std::mutex> mlock(mutex_);
    return queue_.empty();
}


#endif // __cplusplus

#endif // TOOLS_PROFILER_INCLUDE_PROFILEDATAQUEUE_HPP_
