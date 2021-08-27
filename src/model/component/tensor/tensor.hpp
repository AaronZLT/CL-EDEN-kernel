#ifndef SRC_MODE_COMPONENT_TENSOR_TENSOR_HPP_
#define SRC_MODE_COMPONENT_TENSOR_TENSOR_HPP_

#include <vector>
#include <memory>
#include <string>

#include "model/types.hpp"
#include "model/schema/schema_nnc.h"


namespace enn {
namespace model {
namespace component {

class IOperator;

class Tensor {
 private:
    using OperatorVector = std::vector<std::weak_ptr<IOperator>>;

 public:
    using Ptr = std::shared_ptr<Tensor>;

    // This inner class is a Iterator for next_operators_,
    //  which returns iterator of vector<std::shared_ptr<IOperator>> converted from vector<std::weak_ptr<IOperator>>.
    class NextOperatorIterator {
        using LockedOperatorVector = std::vector<std::shared_ptr<IOperator>>;
     private:
        OperatorVector& next_operators_;
        LockedOperatorVector locked_next_ops;
     public:
        NextOperatorIterator(OperatorVector& next_operators)
            : next_operators_{next_operators} {
            for_each(next_operators_.begin(), next_operators_.end(),
                    [&](const std::weak_ptr<IOperator>& op) {
                        locked_next_ops.push_back(op.lock());
                    });
        }
        auto operator[](int index) const { return next_operators_[index]; }
        auto count() const { return next_operators_.size(); }
        typename LockedOperatorVector::const_iterator begin() const {
            return locked_next_ops.cbegin();
        }
        typename LockedOperatorVector::const_iterator end() const {
            return locked_next_ops.cend();
        }
    };

 public:
    virtual ~Tensor() = default;

    const std::string& get_name() const { return name_; }
    int32_t get_id() const { return id_; }
    uint32_t get_data_type() const { return data_type_; }
    const std::vector<uint32_t>& get_shape() const { return shape_; }
    const TFlite::QuantizationParameters* get_quantization_parameters() const { return quantization_parameters_; }
    const TFlite::SymmPerChannelQuantParamters* get_symm_per_channel_quant_parameters() const {
        return symm_per_channel_quant_parameters_;
    }
    // return iterator object
    NextOperatorIterator next() { return NextOperatorIterator{next_operators_}; }
    virtual std::shared_ptr<IOperator> prev() const = 0;
    // Return as bool whether the data in it is constant or not.
    virtual bool is_const() const = 0;

    template <typename Builder, typename TensorType> friend class TensorBuilder;  // Declare Builder class

 protected:
    Tensor()
        : id_{0}, data_type_{TFlite::TensorType_UINT8},
          quantization_parameters_{nullptr}, symm_per_channel_quant_parameters_{nullptr} {}

 protected:
    std::string                           name_;         // Tensor's name
    int32_t                               id_;           // ID to identify each edge(tensor)
    TFlite::TensorType                    data_type_;    // enum TensorType in schema_generated.h
    std::vector<uint32_t>                 shape_;        // Tensor's shape
    const TFlite::QuantizationParameters* quantization_parameters_;
    const TFlite::SymmPerChannelQuantParamters* symm_per_channel_quant_parameters_;
    // Smart pointer to IOperator should be weak_ptr to avoid circular reference.
    OperatorVector next_operators_;  // next operators
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODE_COMPONENT_TENSOR_TENSOR_HPP_