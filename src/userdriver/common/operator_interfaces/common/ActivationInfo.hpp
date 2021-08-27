#pragma once

class ActivationInfo {
 public:
    enum class ActivationType {
        NONE,
        RELU,
        RELU1,
        RELU6,
        TANH,
        SIGMOID
    };
    ActivationInfo() = default;
    ActivationInfo(ActivationType activation_type, const bool& enabled) :
                   activation_type_(activation_type), enabled_(enabled) {
    }
    ActivationType activation() const {
        return activation_type_;
    }

    bool isEnabled() const {
        return enabled_;
    }
    bool disable() {
        enabled_ = false;
        return enabled_;
    }

 private:
    ActivationType activation_type_ = ActivationInfo::ActivationType::NONE;
    bool enabled_ = false;
};
