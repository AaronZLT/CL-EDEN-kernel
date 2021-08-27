#ifndef SRC_MODEL_SCHEMA_NNC_V1_COMPATIBILITY_H_
#define SRC_MODEL_SCHEMA_NNC_V1_COMPATIBILITY_H_

/**
 * Note)
 * NNC version compatibility must be ensured about newly added structs or classes in this header file.
 * It is to prevent the abuse of preprocessor directives(#ifdef) from increasing complexity.
 * Newly added structures or classes are added as a dummy below,
 * and somethings unsupported are notified at runtime with a default value such as nullptr.
 */

namespace tflite {

struct SymmPerChannelQuantParamters {};

}  // namespace tflite

#endif  // SRC_MODEL_SCHEMA_NNC_V1_COMPATIBILITY_H_
