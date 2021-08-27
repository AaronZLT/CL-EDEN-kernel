#include "CLPlatform.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLPlatform::CLPlatform() : num_platforms_(0), num_devices_(0), platforms_(nullptr), devices_(nullptr), selected_platform_(nullptr) {}

CLPlatform::~CLPlatform() {
    free(platforms_);
    free(devices_);
}

std::shared_ptr<CLPlatform> CLPlatform::instance_ = NULL;
std::shared_ptr<CLPlatform> CLPlatform::getInstance() {
    if (instance_ == NULL) {
        instance_ = std::make_shared<CLPlatform>();
    }
    return instance_;
}

Status CLPlatform::initialize() {
    cl_uint err = 0;

    err = clGetPlatformIDs(0, NULL, &num_platforms_);
    DEBUG_PRINT("initializePlatforms(): num_platforms = %d", num_platforms_);

    platforms_ = static_cast<cl_platform_id *>(malloc(sizeof(cl_platform_id) * num_platforms_));
    CHECK_EXPR_RETURN_FAILURE(platforms_ != NULL, "platformsmalloc_failed");

    err = clGetPlatformIDs(num_platforms_, platforms_, NULL);
    CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clGetPlatformIDs() error (2), err: %d", err);

    uint32_t num_arm_platforms = 0;
    uint32_t num_amd_platforms = 0;
    bool arm_exist = false;
    bool amd_exist = false;
    cl_platform_id *selected_arm_platform = nullptr;
    cl_platform_id *selected_amd_platform = nullptr;
    for (cl_uint idx = 0; idx < num_platforms_; idx++) {
        size_t ext_size;
        char *ext_data = NULL;
        err = clGetPlatformInfo(platforms_[idx], CL_PLATFORM_NAME, 0, NULL, &ext_size);
        CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clGetPlatformInfo() error (2), err: %d", err);

        ext_data = static_cast<char *>(malloc(ext_size));

        err = clGetPlatformInfo(platforms_[idx], CL_PLATFORM_NAME, ext_size, ext_data, NULL);
        CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clGetPlatformInfo() error (3), err: %d", err);

        std::string ext_model(ext_data);
        free(ext_data);

        if (std::string::npos != ext_model.find("AMD")) {
            if (num_amd_platforms < 1) {
                num_amd_platforms++;
                selected_amd_platform = &(platforms_[idx]);
                amd_exist = true;
            } else {
                ERROR_PRINT("more than 2 AMD platforms are detected!");
            }
        } else if (std::string::npos != ext_model.find("ARM")) {
            if (num_arm_platforms < 1) {
                num_arm_platforms++;
                selected_arm_platform = &(platforms_[idx]);
                arm_exist = true;
            } else {
                ERROR_PRINT("more than 2 ARM platforms are detected!");
            }
        }
    }

    if (amd_exist) {
        selected_platform_ = selected_amd_platform;
        platform_type_ = PlatformType::AMD;
    } else if(arm_exist) {
        selected_platform_ = selected_arm_platform;
        platform_type_ = PlatformType::ARM;
    }

    if (amd_exist || arm_exist) {
        err = clGetDeviceIDs(*selected_platform_, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices_);
        CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "Error: clGetDeviceIDs() error, err: %d", err);
        DEBUG_PRINT("initializePlatforms(): num_devices = %d", num_devices_);

        devices_ = static_cast<cl_device_id *>(malloc(sizeof(cl_device_id) * num_devices_));
        CHECK_EXPR_RETURN_FAILURE(devices_ != NULL, "devices_malloc_failed");

        err = clGetDeviceIDs(*selected_platform_, CL_DEVICE_TYPE_GPU, num_devices_, devices_, NULL);
        CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clGetDeviceIDs() error, err: %d", err);
    } else {
        ERROR_PRINT("ARM platform is not detected");
    }
    return Status::SUCCESS;
}

Status CLPlatform::validate(const uint32_t &target_device_id) {
    CHECK_EXPR_RETURN_FAILURE(num_devices_ > target_device_id,
                              "CLPlatform::validate() fail (%d > %d)",
                              num_devices_,
                              target_device_id);
    return Status::SUCCESS;
}

cl_uint CLPlatform::getNumDevices() { return num_devices_; }

cl_uint CLPlatform::getNumPlatforms() { return num_platforms_; }

cl_device_id *CLPlatform::getDevices() { return devices_; }

cl_platform_id *CLPlatform::getPlatforms() { return platforms_; }

cl_platform_id *CLPlatform::getPlatform() { return selected_platform_; }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
