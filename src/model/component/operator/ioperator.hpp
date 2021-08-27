#ifndef SRC_MODEL_COMPONENT_OPERATOR_IOPERATOR_HPP_
#define SRC_MODEL_COMPONENT_OPERATOR_IOPERATOR_HPP_

#include <memory>
#include <vector>
#include <string>

#include "model/types.hpp"

namespace enn {
namespace model {
namespace component {

class Tensor;

class IOperator {
 private:
    using TensorList = std::vector<std::shared_ptr<Tensor>>;

 public:
    using Ptr = std::shared_ptr<IOperator>;

    // Iterator for incoming_tensors_ of TensorList
    class InTensorIterator {
     private:
        TensorList& incoming_tensors_;
     public:
        InTensorIterator(TensorList& in_tensors)
            : incoming_tensors_{in_tensors} {}
        auto operator[](int index) { return incoming_tensors_[index]; }
        auto count() { return incoming_tensors_.size(); }
        typename TensorList::const_iterator begin() { return incoming_tensors_.cbegin(); }
        typename TensorList::const_iterator end() { return incoming_tensors_.cend(); }
    } in_tensors;

    // Iterator for outgoing_tensors_ of TensorList
    class OutTensorIterator {
     private:
        TensorList& outgoing_tensors_;
     public:
        OutTensorIterator(TensorList& out_tensors)
            : outgoing_tensors_{out_tensors} {}
        auto operator[](int index) { return outgoing_tensors_[index]; }
        auto count() { return outgoing_tensors_.size(); }
        typename TensorList::const_iterator begin() { return outgoing_tensors_.cbegin(); }
        typename TensorList::const_iterator end() { return outgoing_tensors_.cend(); }
    } out_tensors;

 public:
    virtual ~IOperator() = default;

    const std::string& get_name() const {
        return name_;
    }

 protected:
    template <typename Builder, typename OpType> friend class IOperatorBuilder;  // has creation right of Operator.
    // Inject a reference of feature map list to each iterator when being created.
    IOperator()
        : in_tensors(in_tensor_list_), out_tensors(out_tensor_list_), accelerator_(enn::model::Accelerator::NONE) {}

    std::string               name_;                  // name
    enn::model::Accelerator   accelerator_;           // Will be used to find which UD can excute it
    TensorList                in_tensor_list_;   // incoming tensor list
    TensorList                out_tensor_list_;  // outgoing tensor list
};

};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_COMPONENT_OPERATOR_IOPERATOR_HPP_