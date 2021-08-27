
#ifndef USERDRIVER_NPU_PROFILE_UTIL_H_
#define USERDRIVER_NPU_PROFILE_UTIL_H_

#ifdef EXYNOS_NN_PROFILER

#include <malloc.h>        // malloc

#include "userdriver/unified/vs4l.h"
#include "tool/profiler/include/ExynosNnProfilerApi.h"
#include "common/enn_debug.h"


static inline int _allocate_profiler_struct(struct vs4l_profiler** profiler) {
    // Allocate the vs4l_profiler with profiler level
    *profiler = (struct vs4l_profiler*)malloc(sizeof(struct vs4l_profiler));
    if (*profiler == NULL) {
        ENN_ERR_PRINT_FORCE("Fail on malloc for vs4l_profiler\n");
        return LINK_ALLOCATE_PROFILER_FAILED;
    }
    // TODO(yc18.cho): Set dynamically level after enabling capabilities.
    (*profiler)->level = 1;
    (*profiler)->node = NULL;
    return EDEN_SUCCESS;
}

static inline int _release_profiler_struct(struct vs4l_profiler** profiler) {
    // Release the vs4l_profiler allocated
    free(*profiler);
    return EDEN_SUCCESS;
}

static inline int _allocate_profiler_node(struct vs4l_profiler_node** node, const char* label) {
    // Allocate a vs4l_profiler_node that indicate a scope to be profiled
    // Deleting this memory allocated here is done in ExynosNN Profiler
    *node = (struct vs4l_profiler_node*)malloc(sizeof(struct vs4l_profiler_node));
    if (*node == NULL) {
        ENN_ERR_PRINT_FORCE("Fail on malloc for profiler->node\n");
        return LINK_ALLOCATE_PROFILER_FAILED;
    }
    (*node)->label = (char*)malloc(strlen(label) + 1);
    if ((*node)->label == NULL) {
        ENN_ERR_PRINT_FORCE("Fail on malloc for (*node)->label\n");
         return LINK_ALLOCATE_PROFILER_FAILED;
    }
    if (strlen(label) != strlcpy((*node)->label, label, strlen(label) + 1)) {
        ENN_ERR_PRINT_FORCE("Fail on strcpy for (*node)->label\n");
        return LINK_ALLOCATE_PROFILER_FAILED;
    }
    (*node)->child = NULL;
    return EDEN_SUCCESS;
}

static inline int _allocate_profile_tree(struct vs4l_profiler** profiler) {
    if (_allocate_profiler_struct(profiler) != EDEN_SUCCESS) {
        ENN_ERR_PRINT_FORCE("Failed to allocate the profiler!\n");
        return LINK_ALLOCATE_PROFILER_FAILED;
    }
    if (_allocate_profiler_node(&(*profiler)->node, "Firmware_Execution") != EDEN_SUCCESS) {
        ENN_ERR_PRINT_FORCE("Failed to allocate a node for Firmware_Execution!");
        return LINK_ALLOCATE_PROFILER_FAILED;
    }

    (*profiler)->node->child = (struct vs4l_profiler_node**)malloc(sizeof(struct vs4l_profiler_node*) * 2);
    if ((*profiler)->node->child == NULL) {
        ENN_ERR_PRINT_FORCE("Fail on malloc for (*profiler)->node->child\n");
        return LINK_ALLOCATE_PROFILER_FAILED;
    }
    if (_allocate_profiler_node(&(*profiler)->node->child[0], "Hardware_Execution") != EDEN_SUCCESS) {
        ENN_ERR_PRINT_FORCE("Failed to allocate a node for Hardware_Execution!");
    }
    // A children node containing NULL is needed to tell that there are no more child nodes.
    (*profiler)->node->child[1] = NULL;

    return EDEN_SUCCESS;
}

static inline void _append_profiler_node(struct vs4l_profiler_node* node, uint32_t id) {
    PROFILE_APPEND((struct CalculatedProfileNode*)node, id);
}

#endif  // EXYNOS_NN_PROFILER

#endif  // USERDRIVER_NPU_PROFILE_UTIL_H_