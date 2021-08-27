
#include <android-base/logging.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidl/HidlTransportSupport.h>
#include <hidlmemory/mapping.h>
#include <hwbinder/IPCThreadState.h>
#include <vendor/samsung_slsi/hardware/enn/1.0/IEnnInterface.h>
#include <vendor/samsung_slsi/hardware/enn/1.0/IEnnCallback.h>

/* Android */
using ::android::hidl::memory::V1_0::IMemory;
using ::android::hardware::hidl_vec;
using ::android::hardware::IPCThreadState;

using ::vendor::samsung_slsi::hardware::enn::V1_0::IEnnInterface;
using ::vendor::samsung_slsi::hardware::enn::V1_0::IEnnCallback;
using ::vendor::samsung_slsi::hardware::enn::V1_0::InterfaceBaseInfo;
using ::vendor::samsung_slsi::hardware::enn::V1_0::LoadParameter;
using ::vendor::samsung_slsi::hardware::enn::V1_0::BufferCore;
using ::vendor::samsung_slsi::hardware::enn::V1_0::Region;
using ::vendor::samsung_slsi::hardware::enn::V1_0::Buffer;
using ::vendor::samsung_slsi::hardware::enn::V1_0::DirType;

/* Related to Load/Execution paramters */
using ::vendor::samsung_slsi::hardware::enn::V1_0::InferenceSet;
using ::vendor::samsung_slsi::hardware::enn::V1_0::SessionBufInfo;
using ::vendor::samsung_slsi::hardware::enn::V1_0::InferenceData;
using ::vendor::samsung_slsi::hardware::enn::V1_0::InferenceRegion;
using ::vendor::samsung_slsi::hardware::enn::V1_0::GeneralParameterReturn;

using ::android::hardware::hidl_handle;

namespace enn {
namespace hal {
    using handle = ::android::hardware::hidl_handle;
}
}

struct EnnCallback;
