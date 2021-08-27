/*
 *  osal/include/dmabuf.h
 *
 *   Copyright 2021 Samsung Electronics Co., Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#include "sys/types.h"

#ifndef DMABUF_H_
#define DMABUF_H_

int dmabuf_open();
int dmabuf_close(int dmabuf_fd);
int dmabuf_alloc(int dmabuf_fd, size_t len, unsigned int heap_mask, unsigned int flags);
int dmabuf_free(int dmabuf_fd);

#endif