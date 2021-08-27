#ifndef SRC_MODEL_SCHEMA_SCHEMA_NNC_H_
#define SRC_MODEL_SCHEMA_SCHEMA_NNC_H_

#ifdef SCHEMA_NNC_V1
#include "model/schema/nnc/v1/schema_generated.h"
#include "model/schema/nnc/v1/compatibility.h"
namespace TFlite = tflite;
#else
#include "model/schema/nnc/v2/schema_generated.h"
namespace TFlite = tflite::v2;
#endif

#endif  // SRC_MODEL_SCHEMA_SCHEMA_NNC_H_