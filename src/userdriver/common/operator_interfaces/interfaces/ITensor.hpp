#pragma once

#include <string>
#include <unordered_map>
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#define PLACEHOLDERLENGTH 60
#define ENABLEPLACEHOLDERPRINT true

namespace enn {
namespace ud {

class ITensor {
public:
    virtual size_t getNumOfBytes() = 0;
    virtual DataType getDataType() = 0;
    virtual PrecisionType getPrecisionType() = 0;
    virtual const Dim4 &getDim() = 0;
    virtual const NDims &getDims() = 0;
    virtual uint32_t getDims(const int32_t &idx) = 0;
    virtual uint32_t getNumOfDims() = 0;
    virtual uint32_t getTotalSizeFromDims() = 0;
    virtual DataOrder getDataOrder() = 0;
    virtual float getScale() = 0;
    virtual int32_t getZeroPoint() = 0;
    virtual void setZeroPoint(int32_t zero_point) = 0;
    virtual void setScale(float scale) = 0;
    virtual void set_buffer_index(int32_t index) = 0;
    virtual int32_t get_buffer_index() = 0;
    virtual void set_buffer_type(uint32_t type) = 0;
    virtual uint32_t get_buffer_type() = 0;
    virtual void set_buffer_ptr(void *addr) = 0;
    virtual bool is_const() { return false; }

    virtual Status readData(DataPtr data,
                            bool blocking = true,
                            DataOrderChangeType type = DataOrderChangeType::OTHER,
                            void *event = nullptr) = 0;
    virtual Status writeData(DataPtr data, bool blocking = true, DataOrderChangeType type = DataOrderChangeType::OTHER) = 0;

    virtual Status reconfigure(const PrecisionType &precision,
                               const DataOrder &order,
                               const Dim4 &dims,
                               const float &scale,
                               const int32_t &offset) = 0;
    virtual Status reconfigure(const PrecisionType &precision,
                               const DataOrder &order,
                               const NDims &ndim,
                               const float &scale,
                               const int32_t &offset) = 0;
    virtual Status reconfigureDim(const Dim4 &dims) = 0;
    virtual Status reconfigureDims(const NDims &ndim) = 0;
    virtual Status reconfigureDimAndBuffer(const Dim4 &dims) = 0;
    virtual Status reconfigureDimsAndBuffer(const NDims &dims) = 0;

    virtual Status reorder(const DataOrder &order) = 0;

    virtual void queryTensorInfo() = 0;
    virtual void dumpTensorData(std::string file_name) = 0;
    virtual Status resetInterBuffer() = 0;
    virtual ~ITensor() = default;

    void *raw_buffer = nullptr;
    static void placeholder(bool msgb, std::string name, ColorType color = ColorType::WHITE, bool thin = false) {
        std::string msg = name + " is " + (msgb ? "true" : "false");
        color = msgb ? ColorType::GREEN : ColorType::RED;
        placeholderprint(msg, color, thin);
    }
    static void placeholder(std::string msg, ColorType color = ColorType::WHITE, bool thin = false) {
        placeholderprint(msg, color, thin);
    }
    static void placeholder(std::string msgs, int msgi, ColorType color = ColorType::WHITE, bool thin = false) {
        placeholderprint(msgs + " → " + std::to_string(msgi), color, thin);
    }
    static void placeholder(int msg, ColorType color = ColorType::WHITE, bool thin = false) {
        placeholderprint(std::to_string(msg), color, thin);
    }

private:
    static void placeholderprint(std::string msg, ColorType color, bool thin) {
        if (ENABLEPLACEHOLDERPRINT) {
            std::string PlaceHolderColors[17] = {
                "\033[0m",         /* Reset */
                "\033[30m",        /* Black */
                "\033[31m",        /* Red */
                "\033[32m",        /* Green */
                "\033[33m",        /* Yellow */
                "\033[34m",        /* Blue */
                "\033[35m",        /* Magenta */
                "\033[36m",        /* Cyan */
                "\033[37m",        /* White */
                "\033[1m\033[30m", /* Bold Black */
                "\033[1m\033[31m", /* Bold Red */
                "\033[1m\033[32m", /* Bold Green */
                "\033[1m\033[33m", /* Bold Yellow */
                "\033[1m\033[34m", /* Bold Blue */
                "\033[1m\033[35m", /* Bold Magenta */
                "\033[1m\033[36m", /* Bold Cyan */
                "\033[1m\033[37m", /* Bold White */
            };
            if (msg.length() != 0) {
                msg = " " + msg + " ";
            }
            int i = msg.length(), j = 0;
            int x = (PLACEHOLDERLENGTH - i) / 2;
            if (!thin)
                std::cout << std::endl;
            for (j = 0; j < x; j++) {
                std::cout << "-";
            }
            if (msg.length() != 0) {
                std::cout << PlaceHolderColors[(int)color] << msg << PlaceHolderColors[(int)ColorType::WHITE];
            }
            for (j += i; j < PLACEHOLDERLENGTH; j++) {
                std::cout << "-";
            }
            std::cout << std::endl;
        }
    }

};  // class ITensor

}  // namespace ud
}  // namespace enn
